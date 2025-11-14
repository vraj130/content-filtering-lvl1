#!/usr/bin/env python3
"""
Test script for instruction-tuned Gemma model on multiple datasets
Tests both in-distribution and out-of-distribution data
"""

import os
import sys
import yaml
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


class InstructionModelTester:
    """Test instruction-tuned model on multiple datasets"""
    
    def __init__(self, config_path, model_path):
        """
        Initialize tester
        
        Args:
            config_path: Path to training config YAML
            model_path: Path to trained model directory (final_model)
        """
        self.config_path = config_path
        self.model_path = model_path
        self.config = self.load_config()
        self.device = None
        self.tokenizer = None
        self.model = None
        self.results = {}
        
    def load_config(self):
        """Load configuration from YAML"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_device(self):
        """Setup GPU"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['gpu']['graphic_card']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def load_model(self):
        """Load trained model and tokenizer"""
        print(f"\n🤖 Loading model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side='right'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Tokenizer loaded")
        
        # Load base model with quantization
        base_model_id = self.config['model']['hugging_face_model_id']
        print(f"Loading base model: {base_model_id}")
        
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
            torch_dtype=torch.bfloat16,
        )
        
        # Load LoRA adapter
        print("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        
        print("✅ Model loaded successfully")
    
    def create_prompt(self, text):
        """Create instruction prompt (same as training)"""
        prompt_template = self.config['instruction']['prompt_template']
        prompt = prompt_template.format(text=text)
        return prompt
    
    def predict_single(self, text):
        """
        Predict class for a single text
        
        Returns:
            predicted_class (0 or 1), confidence (float), answer_token (str)
        """
        # Create prompt
        prompt = self.create_prompt(text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(self.device)
        
        # Get token IDs for Yes/No
        yes_tokens = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_tokens = self.tokenizer.encode("No", add_special_tokens=False)
        yes_token_id = yes_tokens[0]
        no_token_id = no_tokens[0]
        
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
        yes_score = first_token_logits[yes_token_id].item()
        no_score = first_token_logits[no_token_id].item()
        
        # Calculate probabilities
        probs = F.softmax(torch.tensor([no_score, yes_score]), dim=0)
        yes_prob = probs[1].item()
        
        # Predict
        predicted_class = 1 if yes_score > no_score else 0
        confidence = yes_prob if predicted_class == 1 else (1 - yes_prob)
        answer_token = "Yes" if predicted_class == 1 else "No"
        
        return predicted_class, confidence, answer_token
    
    def test_dataset(self, parquet_path, dataset_name):
        """
        Test model on a single parquet file
        
        Args:
            parquet_path: Path to parquet file
            dataset_name: Name for this dataset (for reporting)
            
        Returns:
            dict: Comprehensive metrics
        """
        print(f"\n{'='*80}")
        print(f"Testing: {dataset_name}")
        print(f"File: {parquet_path}")
        print(f"{'='*80}")
        
        # Load data
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df):,} samples")
        
        # Check true labels
        true_labels = df['label'].values
        unique_labels = np.unique(true_labels)
        print(f"True labels in dataset: {unique_labels}")
        
        if len(unique_labels) == 1:
            true_label = unique_labels[0]
            label_name = "AI" if true_label == 1 else "Non-AI"
            print(f"✅ Homogeneous dataset: All samples are '{label_name}' (label={true_label})")
        else:
            print(f"⚠️  Mixed dataset: Contains multiple labels {unique_labels}")
            true_label = None
        
        # Run predictions
        predictions = []
        confidences = []
        answer_tokens = []
        
        print(f"\n🔮 Running predictions...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            pred_class, conf, answer = self.predict_single(row['text'])
            predictions.append(pred_class)
            confidences.append(conf)
            answer_tokens.append(answer)
        
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Calculate metrics
        print(f"\n📊 Computing metrics...")
        
        # Basic accuracy
        accuracy = accuracy_score(true_labels, predictions)
        
        # Prediction distribution
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        pred_0_count = pred_counts.get(0, 0)
        pred_1_count = pred_counts.get(1, 0)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, labels=[0, 1], zero_division=0
        )
        
        # ROC-AUC (if applicable)
        try:
            roc_auc = roc_auc_score(true_labels, confidences)
        except:
            roc_auc = None
        
        # Confidence analysis
        correct_mask = (predictions == true_labels)
        incorrect_mask = ~correct_mask
        
        mean_conf_overall = np.mean(confidences)
        mean_conf_correct = np.mean(confidences[correct_mask]) if correct_mask.any() else 0.0
        mean_conf_incorrect = np.mean(confidences[incorrect_mask]) if incorrect_mask.any() else 0.0
        
        # Class-specific metrics (for homogeneous datasets)
        if true_label is not None:
            if true_label == 0:  # Non-AI dataset
                true_negatives = cm[0, 0]
                false_positives = cm[0, 1]
                class_recall = recall[0]  # Recall for non-AI
                error_rate = false_positives / len(df)
            else:  # AI dataset
                true_positives = cm[1, 1]
                false_negatives = cm[1, 0]
                class_recall = recall[1]  # Recall for AI
                error_rate = false_negatives / len(df)
        else:
            class_recall = None
            error_rate = None
        
        # Compile results
        results = {
            'dataset_name': dataset_name,
            'file_path': parquet_path,
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(df),
            'true_label': int(true_label) if true_label is not None else 'mixed',
            'true_label_name': label_name if true_label is not None else 'mixed',
            
            # Overall metrics
            'accuracy': float(accuracy),
            'error_rate': float(error_rate) if error_rate is not None else None,
            
            # Prediction distribution
            'predicted_non_ai_count': int(pred_0_count),
            'predicted_ai_count': int(pred_1_count),
            'predicted_non_ai_pct': float(pred_0_count / len(df) * 100),
            'predicted_ai_pct': float(pred_1_count / len(df) * 100),
            
            # Confusion matrix
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1]),
            
            # Per-class metrics
            'non_ai_precision': float(precision[0]),
            'non_ai_recall': float(recall[0]),
            'non_ai_f1': float(f1[0]),
            'ai_precision': float(precision[1]),
            'ai_recall': float(recall[1]),
            'ai_f1': float(f1[1]),
            
            # ROC-AUC
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            
            # Confidence analysis
            'mean_confidence_overall': float(mean_conf_overall),
            'mean_confidence_correct': float(mean_conf_correct),
            'mean_confidence_incorrect': float(mean_conf_incorrect),
            'correct_predictions': int(correct_mask.sum()),
            'incorrect_predictions': int(incorrect_mask.sum()),
            
            # For homogeneous datasets
            'class_recall': float(class_recall) if class_recall is not None else None,
        }
        
        # Print summary
        self.print_results(results)
        
        return results
    
    def print_results(self, results):
        """Pretty print results"""
        print(f"\n{'='*80}")
        print(f"RESULTS: {results['dataset_name']}")
        print(f"{'='*80}")
        
        print(f"\n📋 Dataset Info:")
        print(f"   Samples: {results['sample_count']:,}")
        print(f"   True label: {results['true_label_name']} (label={results['true_label']})")
        
        print(f"\n🎯 Overall Performance:")
        print(f"   Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        if results['error_rate'] is not None:
            print(f"   Error Rate: {results['error_rate']:.4f} ({results['error_rate']*100:.2f}%)")
        
        print(f"\n🔮 Prediction Distribution:")
        print(f"   Predicted Non-AI: {results['predicted_non_ai_count']:,} ({results['predicted_non_ai_pct']:.1f}%)")
        print(f"   Predicted AI: {results['predicted_ai_count']:,} ({results['predicted_ai_pct']:.1f}%)")
        
        print(f"\n📊 Confusion Matrix:")
        print(f"                Predicted")
        print(f"                Non-AI    AI")
        print(f"   Actual Non-AI  {results['true_negatives']:<9} {results['false_positives']}")
        print(f"   Actual AI      {results['false_negatives']:<9} {results['true_positives']}")
        
        print(f"\n📈 Per-Class Metrics:")
        print(f"   Non-AI: Precision={results['non_ai_precision']:.4f}, Recall={results['non_ai_recall']:.4f}, F1={results['non_ai_f1']:.4f}")
        print(f"   AI:     Precision={results['ai_precision']:.4f}, Recall={results['ai_recall']:.4f}, F1={results['ai_f1']:.4f}")
        
        if results['roc_auc'] is not None:
            print(f"\n🎲 ROC-AUC: {results['roc_auc']:.4f}")
        
        print(f"\n💯 Confidence Analysis:")
        print(f"   Mean (overall): {results['mean_confidence_overall']:.4f}")
        print(f"   Mean (correct): {results['mean_confidence_correct']:.4f}")
        print(f"   Mean (incorrect): {results['mean_confidence_incorrect']:.4f}")
        print(f"   Correct: {results['correct_predictions']:,}/{results['sample_count']:,}")
        print(f"   Incorrect: {results['incorrect_predictions']:,}/{results['sample_count']:,}")
        
        if results['class_recall'] is not None:
            print(f"\n✅ Class-specific Recall: {results['class_recall']:.4f}")
    
    def save_results(self, output_dir):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual results as JSON
        individual_dir = os.path.join(output_dir, "individual_results")
        os.makedirs(individual_dir, exist_ok=True)
        
        for dataset_name, result in self.results.items():
            filename = f"{dataset_name.replace(' ', '_').replace('/', '_')}_{timestamp}.json"
            filepath = os.path.join(individual_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"   💾 {filepath}")
        
        # Save summary CSV
        summary_df = pd.DataFrame([
            {
                'dataset': r['dataset_name'],
                'samples': r['sample_count'],
                'true_label': r['true_label_name'],
                'accuracy': r['accuracy'],
                'error_rate': r['error_rate'],
                'ai_recall': r['ai_recall'],
                'ai_precision': r['ai_precision'],
                'non_ai_recall': r['non_ai_recall'],
                'non_ai_precision': r['non_ai_precision'],
                'roc_auc': r['roc_auc'],
                'mean_confidence': r['mean_confidence_overall'],
            }
            for r in self.results.values()
        ])
        
        summary_csv = os.path.join(output_dir, f"summary_report_{timestamp}.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"   📊 {summary_csv}")
        
        # Save complete results as JSON
        complete_json = os.path.join(output_dir, f"complete_results_{timestamp}.json")
        with open(complete_json, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"   📄 {complete_json}")
        
        print(f"\n✅ All results saved to: {output_dir}") 
    
    def run_tests(self, test_configs):
        """
        Run tests on multiple datasets
        
        Args:
            test_configs: List of dicts with 'path' and 'name' keys
        """
        print("\n" + "="*80)
        print("INSTRUCTION MODEL TESTING")
        print("="*80)
        print(f"Model: {self.model_path}")
        print(f"Config: {self.config_path}")
        print(f"Number of test datasets: {len(test_configs)}")
        
        # Setup
        self.setup_device()
        self.load_model()
        
        # Test each dataset
        for config in test_configs:
            result = self.test_dataset(config['path'], config['name'])
            self.results[config['name']] = result
        
        # Save results
        print(f"\n{'='*80}")
        print("Saving results...")
        print(f"{'='*80}")
        self.save_results("outputs/test_results")
        
        print(f"\n{'='*80}")
        print("✅ ALL TESTING COMPLETE!")
        print(f"{'='*80}")


def main():
    """Main entry point"""
    # Configuration
    config_path = "config/instruction_training_config.yaml"
    model_path = "outputs/checkpoints/gemma3_20k_balanced_it/final_model"
    
    # Define test datasets
    test_configs = [
        # In-distribution tests
        {
            'name': 'in_dist_negative',
            'path': '/workspace/content-filtering-lvl1/data/final_data_4/in_distribution_test_data/alignment_dataset_ai_negative_8k.parquet'
        },
        {
            'name': 'in_dist_positive',
            'path': '/workspace/content-filtering-lvl1/data/final_data_4/in_distribution_test_data/alignment_dataset_ai_positive_8k.parquet'
        },
        # Out-of-distribution tests (add the 3 files here)
        # Uncomment and update with actual filenames:
        # {
        #     'name': 'ood_dataset_1',
        #     'path': '/workspace/content-filtering-lvl1/data/final_data_4/ood_distribution_test_data/file1.parquet'
        # },
    ]
    
    # # Check which OOD files exist
    # ood_dir = "/workspace/content-filtering-lvl1/data/final_data_4/ood_distribution_test_data"
    # if os.path.exists(ood_dir):
    #     ood_files = [f for f in os.listdir(ood_dir) if f.endswith('.parquet')]
    #     print(f"\n📂 Found {len(ood_files)} OOD test files:")
    #     for f in ood_files:
    #         print(f"   - {f}")
    #         test_configs.append({
    #             'name': f'ood_{f.replace(".parquet", "")}',
    #             'path': os.path.join(ood_dir, f)
    #         })
    
    # Run tests
    tester = InstructionModelTester(config_path, model_path)
    tester.run_tests(test_configs)


if __name__ == "__main__":
    main()
