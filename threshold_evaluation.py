#!/usr/bin/env python3
"""
Unified Threshold Evaluation Script
Evaluates Gemma and ModernBERT models across multiple thresholds (0.1-0.9)
on multiple parquet datasets with comprehensive metrics.
"""

import os
import sys
import json
import yaml
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
import re
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# CONFIGURATION (Hardcoded - Modify these as needed)
# ============================================================================

# Model paths
MODEL_PATHS = {
    'gemma': 'outputs/checkpoints/gemma-v2-chat-template-1117/final_model',
    'modernbert': 'outputs/checkpoints/modernbert_large_1116/final_model'
}

# Config paths
CONFIG_PATHS = {
    'gemma': 'config/instruction_training_config.yaml',
    'modernbert': 'config/modernbert_training_config.yaml'
}

# Input parquet files (modify these to your datasets)
INPUT_PARQUETS = [
    'data/content_filtering_extensive/alignment_dataset_ai_positive_v3_all.parquet',
    'data/content_filtering_extensive/arxiv_papers_ai_positive_all.parquet',
    'data/content_filtering_extensive/fineweb_ai_negative_v3_all.parquet',
    'data/content_filtering_extensive/fineweb_ai_positive_v3_all.parquet'
]

# Thresholds to evaluate
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Default batch size (can be overridden with --batch-size)
DEFAULT_BATCH_SIZE = 16

# Output directory
OUTPUT_DIR = 'outputs/threshold_evaluation'

# ============================================================================
# Gemma Predictor Class
# ============================================================================

class GemmaPredictor:
    """Predictor for Gemma instruction-tuned model with chat template"""
    
    def __init__(self, config_path, model_path):
        self.config_path = config_path
        self.model_path = model_path
        self.config = self.load_config()
        self.device = None
        self.tokenizer = None
        self.model = None
        self.yes_token_id = None
        self.no_token_id = None
        
    def load_config(self):
        """Load configuration from YAML"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_device(self):
        """Setup GPU"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['gpu']['graphic_card']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
    
    def load_model(self):
        """Load trained Gemma model with LoRA adapter"""
        print(f"\n🤖 Loading Gemma model from: {self.model_path}")
        
        # Load tokenizer with left padding for decoder-only models (required for batched generation)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side='left',
            local_files_only=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Tokenizer loaded")
        
        # Get Yes/No token IDs
        yes_tokens = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_tokens = self.tokenizer.encode("No", add_special_tokens=False)
        self.yes_token_id = yes_tokens[0]
        self.no_token_id = no_tokens[0]
        print(f"   Yes token ID: {self.yes_token_id}, No token ID: {self.no_token_id}")
        
        # Load base model with quantization
        base_model_id = self.config['model']['hugging_face_model_id']
        print(f"🔧 Loading base model: {base_model_id}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config['quantization']['load_in_4bit'],
            bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=getattr(torch, self.config['quantization']['bnb_4bit_compute_dtype'])
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        # Load LoRA adapter
        print("🔌 Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        
        print("✅ Gemma model loaded successfully\n")
    
    def create_prompt(self, text):
        """
        Create instruction prompt using Gemma's official chat template
        CRITICAL: Uses tokenizer.apply_chat_template() for proper formatting
        """
        # Create user message with instruction
        user_message = self.config['instruction']['prompt_template'].format(text=text)
        
        # Inference: only user message, add generation prompt
        messages = [{"role": "user", "content": user_message}]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return prompt
    
    def predict_single(self, text):
        """
        Predict AI probability for a single text
        Returns raw probability (not threshold-based prediction)
        
        Returns:
            ai_probability (float): Probability of AI class
        """
        # Create prompt with chat template
        prompt = self.create_prompt(text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False
            )
        
        # Get logits for Yes/No tokens
        first_token_logits = outputs.scores[0][0]
        yes_score = first_token_logits[self.yes_token_id].item()
        no_score = first_token_logits[self.no_token_id].item()
        
        # Calculate probabilities
        probs = F.softmax(torch.tensor([no_score, yes_score]), dim=0)
        ai_probability = probs[1].item()  # Probability of AI class (Yes)
        
        return ai_probability
    
    def predict_batch(self, texts):
        """
        Predict AI probabilities for a batch of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of AI probabilities (floats)
        """
        # Create prompts for all texts using chat template
        prompts = [self.create_prompt(text) for text in texts]
        
        # Tokenize batch with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
            padding=True
        ).to(self.device)
        
        # Generate for batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False
            )
        
        # Extract logits for Yes/No tokens for each sample in batch
        # outputs.scores[0] has shape [batch_size, vocab_size]
        first_token_logits = outputs.scores[0]  # Shape: [batch_size, vocab_size]
        
        ai_probabilities = []
        for i in range(len(texts)):
            # Get Yes/No scores for this sample
            yes_score = first_token_logits[i, self.yes_token_id].item()
            no_score = first_token_logits[i, self.no_token_id].item()
            
            # Calculate probability
            probs = F.softmax(torch.tensor([no_score, yes_score]), dim=0)
            ai_probability = probs[1].item()  # Probability of AI class (Yes)
            ai_probabilities.append(ai_probability)
        
        return ai_probabilities


# ============================================================================
# ModernBERT Predictor Class
# ============================================================================

class ModernBERTPredictor:
    """Predictor for ModernBERT classification model"""
    
    def __init__(self, config_path, model_path):
        self.config_path = config_path
        self.model_path = model_path
        self.config = self.load_config()
        self.device = None
        self.tokenizer = None
        self.model = None
        
    def load_config(self):
        """Load configuration from YAML"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_device(self):
        """Setup GPU"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['gpu']['graphic_card']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
    
    def load_model(self):
        """Load trained ModernBERT classification model"""
        print(f"\n🤖 Loading ModernBERT model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side='right',
            local_files_only=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Tokenizer loaded")
        
        # Load classification model
        print(f"🔧 Loading ModernBERT classification model...")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.config['training']['bf16'] else torch.float32,
            device_map="auto",
            local_files_only=True
        )
        
        self.model.eval()
        
        print("✅ ModernBERT model loaded successfully")
        print(f"   Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Number of labels: {self.model.config.num_labels}\n")
    
    def predict_single(self, text):
        """
        Predict AI probability for a single text
        Returns raw probability (not threshold-based prediction)
        
        Returns:
            ai_probability (float): Probability of AI class
        """
        # Tokenize (no template needed for ModernBERT)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['model']['max_length']
        ).to(self.device)
        
        # Get logits from classification head
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # Shape: [2]
        
        # Calculate probabilities
        probs = F.softmax(logits, dim=0)
        ai_probability = probs[1].item()  # Probability of AI class (index 1)
        
        return ai_probability
    
    def predict_batch(self, texts):
        """
        Predict AI probabilities for a batch of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of AI probabilities (floats)
        """
        # Tokenize batch with padding (no template needed for ModernBERT)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['model']['max_length'],
            padding=True
        ).to(self.device)
        
        # Get logits from classification head
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: [batch_size, 2]
        
        # Calculate probabilities for each sample
        probs = F.softmax(logits, dim=1)  # Shape: [batch_size, 2]
        ai_probabilities = probs[:, 1].cpu().tolist()  # Extract AI class probabilities
        
        return ai_probabilities


# ============================================================================
# Prediction Caching Function
# ============================================================================

def predict_and_cache(predictor, parquet_files, batch_size=16):
    """
    Run predictions once on all datasets and cache results
    
    Args:
        predictor: GemmaPredictor or ModernBERTPredictor instance
        parquet_files: List of parquet file paths
        batch_size: Batch size for inference (default: 16)
        
    Returns:
        DataFrame with columns: text, true_label, ai_probability, dataset_name
    """
    print("\n" + "="*80)
    print("PREDICTION CACHING PHASE")
    print("="*80)
    print(f"Batch size: {batch_size}")
    
    all_results = []
    
    for parquet_path in parquet_files:
        print(f"\n📂 Processing: {parquet_path}")
        
        # Load parquet file
        df = pd.read_parquet(parquet_path)
        dataset_name = Path(parquet_path).name
        
        print(f"   Loaded {len(df):,} samples")
        
        # Check if labels exist
        has_labels = 'label' in df.columns
        if not has_labels:
            print(f"   ⚠️  WARNING: No labels found in {dataset_name}")
            continue
        
        # Collect all texts and labels
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc=f"   Predicting {dataset_name}"):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Get predictions for batch
            batch_probs = predictor.predict_batch(batch_texts)
            
            # Store results
            for text, label, prob in zip(batch_texts, batch_labels, batch_probs):
                all_results.append({
                    'text': text,
                    'true_label': label,
                    'ai_probability': prob,
                    'dataset_name': dataset_name
                })
    
    # Create cached DataFrame
    cached_df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("CACHING COMPLETE")
    print("="*80)
    print(f"Total samples cached: {len(cached_df):,}")
    print(f"Datasets: {cached_df['dataset_name'].nunique()}")
    print("="*80)
    
    return cached_df


# ============================================================================
# Threshold Evaluation Function
# ============================================================================

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive metrics"""
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    
    # Confidence statistics
    correct_mask = np.array(y_true) == np.array(y_pred)
    if correct_mask.sum() > 0:
        mean_confidence_correct = np.mean([y_prob[i] if y_pred[i] == 1 else (1 - y_prob[i]) 
                                           for i in range(len(y_pred)) if correct_mask[i]])
    else:
        mean_confidence_correct = 0.0
    
    if (~correct_mask).sum() > 0:
        mean_confidence_incorrect = np.mean([y_prob[i] if y_pred[i] == 1 else (1 - y_prob[i]) 
                                             for i in range(len(y_pred)) if not correct_mask[i]])
    else:
        mean_confidence_incorrect = 0.0
    
    mean_confidence_overall = np.mean([y_prob[i] if y_pred[i] == 1 else (1 - y_prob[i]) 
                                       for i in range(len(y_pred))])
    
    return {
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'accuracy': float(accuracy),
        'nonai_precision': float(precision[0]),
        'nonai_recall': float(recall[0]),
        'nonai_f1': float(f1[0]),
        'ai_precision': float(precision[1]),
        'ai_recall': float(recall[1]),
        'ai_f1': float(f1[1]),
        'mean_confidence_correct': float(mean_confidence_correct),
        'mean_confidence_incorrect': float(mean_confidence_incorrect),
        'mean_confidence_overall': float(mean_confidence_overall)
    }


def evaluate_thresholds(cached_df, thresholds):
    """
    Evaluate performance across multiple thresholds
    
    Args:
        cached_df: DataFrame with cached predictions
        thresholds: List of thresholds to evaluate
        
    Returns:
        Dictionary with results for each threshold
    """
    print("\n" + "="*80)
    print("THRESHOLD EVALUATION PHASE")
    print("="*80)
    
    results = {}
    
    # Calculate ROC-AUC once (threshold-independent)
    y_true_combined = cached_df['true_label'].values
    y_prob_combined = cached_df['ai_probability'].values
    
    try:
        roc_auc_combined = roc_auc_score(y_true_combined, y_prob_combined)
        print(f"\n📊 Combined ROC-AUC: {roc_auc_combined:.4f}")
    except Exception as e:
        print(f"\n⚠️  Could not calculate ROC-AUC: {e}")
        roc_auc_combined = None
    
    # Calculate per-dataset ROC-AUC
    per_dataset_roc_auc = {}
    for dataset_name in cached_df['dataset_name'].unique():
        dataset_df = cached_df[cached_df['dataset_name'] == dataset_name]
        try:
            roc_auc = roc_auc_score(dataset_df['true_label'].values, dataset_df['ai_probability'].values)
            per_dataset_roc_auc[dataset_name] = float(roc_auc)
        except Exception as e:
            per_dataset_roc_auc[dataset_name] = None
    
    # Evaluate each threshold
    for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
        print(f"\n📏 Threshold: {threshold}")
        
        # Apply threshold to get predictions
        cached_df['predicted_label'] = (cached_df['ai_probability'] >= threshold).astype(int)
        
        # Combined metrics
        y_true = cached_df['true_label'].values
        y_pred = cached_df['predicted_label'].values
        y_prob = cached_df['ai_probability'].values
        
        combined_metrics = calculate_metrics(y_true, y_pred, y_prob)
        print(f"   Combined Accuracy: {combined_metrics['accuracy']:.4f}")
        
        # Per-dataset metrics
        per_dataset_metrics = {}
        for dataset_name in cached_df['dataset_name'].unique():
            dataset_df = cached_df[cached_df['dataset_name'] == dataset_name]
            
            y_true_ds = dataset_df['true_label'].values
            y_pred_ds = dataset_df['predicted_label'].values
            y_prob_ds = dataset_df['ai_probability'].values
            
            dataset_metrics = calculate_metrics(y_true_ds, y_pred_ds, y_prob_ds)
            per_dataset_metrics[dataset_name] = dataset_metrics
            
            print(f"   {dataset_name}: Accuracy = {dataset_metrics['accuracy']:.4f}")
        
        # Store results
        results[str(threshold)] = {
            'combined': combined_metrics,
            'per_dataset': per_dataset_metrics
        }
    
    print("\n" + "="*80)
    print("THRESHOLD EVALUATION COMPLETE")
    print("="*80)
    
    return results, roc_auc_combined, per_dataset_roc_auc


# ============================================================================
# JSON Output Function
# ============================================================================

def save_results_to_json(model_type, model_path, datasets, thresholds, results, 
                         roc_auc_combined, per_dataset_roc_auc, output_dir):
    """
    Save evaluation results to JSON file
    
    Args:
        model_type: 'gemma' or 'modernbert'
        model_path: Path to model
        datasets: List of dataset names
        thresholds: List of thresholds evaluated
        results: Dictionary with threshold results
        roc_auc_combined: Combined ROC-AUC score
        per_dataset_roc_auc: Per-dataset ROC-AUC scores
        output_dir: Output directory
        
    Returns:
        Path to saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{model_type}_threshold_eval_{timestamp}.json")
    
    output_data = {
        'model_type': model_type,
        'model_path': model_path,
        'datasets': datasets,
        'thresholds': thresholds,
        'roc_auc_combined': roc_auc_combined,
        'per_dataset_roc_auc': per_dataset_roc_auc,
        'results': results,
        'evaluation_timestamp': timestamp
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return output_file


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main entry point"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate model performance across multiple thresholds'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['gemma', 'modernbert'],
        required=True,
        help='Model(s) to evaluate: gemma, modernbert, or both'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Batch size for inference (default: {DEFAULT_BATCH_SIZE})'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("UNIFIED THRESHOLD EVALUATION")
    print("="*80)
    print(f"Models to evaluate: {', '.join(args.models)}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Datasets: {len(INPUT_PARQUETS)}")
    for dataset in INPUT_PARQUETS:
        print(f"  - {dataset}")
    print("="*80)
    
    # Evaluate each model
    for model_type in args.models:
        print(f"\n{'='*80}")
        print(f"EVALUATING: {model_type.upper()}")
        print(f"{'='*80}")
        
        # Get paths
        model_path = MODEL_PATHS[model_type]
        config_path = CONFIG_PATHS[model_type]
        
        print(f"Model path: {model_path}")
        print(f"Config path: {config_path}")
        
        # Initialize predictor
        if model_type == 'gemma':
            predictor = GemmaPredictor(config_path, model_path)
        else:  # modernbert
            predictor = ModernBERTPredictor(config_path, model_path)
        
        predictor.setup_device()
        predictor.load_model()
        
        # Predict and cache
        cached_df = predict_and_cache(predictor, INPUT_PARQUETS, args.batch_size)
        
        # Evaluate thresholds
        results, roc_auc_combined, per_dataset_roc_auc = evaluate_thresholds(cached_df, THRESHOLDS)
        
        # Save results
        dataset_names = [Path(p).name for p in INPUT_PARQUETS]
        output_file = save_results_to_json(
            model_type=model_type,
            model_path=model_path,
            datasets=dataset_names,
            thresholds=THRESHOLDS,
            results=results,
            roc_auc_combined=roc_auc_combined,
            per_dataset_roc_auc=per_dataset_roc_auc,
            output_dir=OUTPUT_DIR
        )
        
        print(f"\n✅ {model_type.upper()} evaluation complete!")
        print(f"   Results: {output_file}")
    
    print("\n" + "="*80)
    print("ALL EVALUATIONS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

