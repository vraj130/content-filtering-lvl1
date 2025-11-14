#!/usr/bin/env python3
"""
Single Threshold Evaluation Script for Instruction-Tuned Gemma Model
Evaluates model performance at a specified threshold
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
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


class ThresholdEvaluator:
    """Evaluate classification performance at a single threshold"""
    
    def __init__(self, config_path, model_path, data_name, threshold=0.5):
        self.config_path = config_path
        self.model_path = model_path
        self.data_name = data_name
        self.threshold = threshold
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
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        # Load LoRA adapter
        print("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        # self.model = base_model
        self.model.eval()
        
        print("✅ Model loaded successfully")
    
    def create_prompt(self, text):
        """Create instruction prompt (same as training)"""
        prompt_template = self.config['instruction']['prompt_template']
        text = re.sub(r'\n', ' ', text.strip())     
        prompt = prompt_template.format(text=text)
        return prompt
    
    def predict_single(self, text):
        """
        Predict class for a single text, return AI probability
        
        Returns:
            yes_prob (float): Probability that text is AI-generated
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
        
        # Calculate AI probability
        probs = F.softmax(torch.tensor([no_score, yes_score]), dim=0)
        yes_prob = probs[1].item()
        
        return yes_prob
    
    def get_predictions(self, parquet_paths, dataset_names):
        """
        Get AI probabilities for all samples in datasets
        
        Returns:
            dict: {dataset_name: {'true_labels': array, 'ai_probs': array, 'texts': array}}
        """
        results = {}
        
        for parquet_path, dataset_name in zip(parquet_paths, dataset_names):
            print(f"\n{'='*80}")
            print(f"Loading predictions for: {dataset_name}")
            print(f"{'='*80}")
            
            # Load data
            df = pd.read_parquet(parquet_path)
            print(f"Loaded {len(df):,} samples")
            
            true_labels = df['label'].values
            texts = df['text'].values
            unique_labels = np.unique(true_labels)
            print(f"True labels: {unique_labels}")
            
            # Get AI probabilities
            ai_probs = []
            print(f"Getting AI probabilities...")
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
                yes_prob = self.predict_single(row['text'])
                ai_probs.append(yes_prob)
            
            ai_probs = np.array(ai_probs)
            
            results[dataset_name] = {
                'true_labels': true_labels,
                'ai_probs': ai_probs,
                'texts': texts,
                'sample_count': len(df)
            }
            
            print(f"✅ Complete - AI prob range: [{ai_probs.min():.3f}, {ai_probs.max():.3f}]")
            print(f"   Mean AI prob: {ai_probs.mean():.3f}")
        
        return results
    
    def evaluate_threshold(self, true_labels, ai_probs, threshold):
        """Evaluate metrics at a specific threshold"""
        predictions = (ai_probs >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, labels=[0, 1], zero_division=0
        )
        cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'ai_precision': precision[1],
            'ai_recall': recall[1],
            'ai_f1': f1[1],
            'nonai_precision': precision[0],
            'nonai_recall': recall[0],
            'nonai_f1': f1[0],
            'true_negatives': cm[0, 0],
            'false_positives': cm[0, 1],
            'false_negatives': cm[1, 0],
            'true_positives': cm[1, 1],
        }
    
    def evaluate_all_datasets(self, predictions_dict):
        """
        Evaluate all datasets at the specified threshold
        
        Args:
            predictions_dict: Dict from get_predictions()
        
        Returns:
            dict: Evaluation results for each dataset
        """
        print(f"\n{'='*80}")
        print(f"THRESHOLD EVALUATION")
        print(f"{'='*80}")
        print(f"Evaluating at threshold: {self.threshold:.3f}")
        
        results = {}
        
        for dataset_name, data in predictions_dict.items():
            print(f"\n\n{'='*80}")
            print(f"Evaluating: {dataset_name}")
            print(f"{'='*80}")
            
            true_labels = data['true_labels']
            ai_probs = data['ai_probs']
            texts = data['texts']
            
            # Evaluate at the specified threshold
            metrics = self.evaluate_threshold(true_labels, ai_probs, self.threshold)
            
            # Get predictions for classification report
            predictions = (ai_probs >= self.threshold).astype(int)
            class_report = classification_report(
                true_labels, 
                predictions, 
                target_names=['Non-AI', 'AI'],
                digits=4
            )
            
            results[dataset_name] = {
                'metrics': metrics,
                'classification_report': class_report,
                'sample_count': data['sample_count'],
                'texts': texts,
                'true_labels': true_labels,
                'predicted_labels': predictions,
                'ai_probs': ai_probs
            }
            
            # Print results
            self.print_evaluation_results(dataset_name, metrics, class_report)
        
        return results
    
    def print_evaluation_results(self, dataset_name, metrics, class_report):
        """Pretty print evaluation results"""
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS: {dataset_name}")
        print(f"{'='*80}")
        
        # Human-readable summary section
        print(f"\n📋 Results Summary for: {dataset_name}")
        print("━" * 80)
        
        total_nonai = metrics['true_negatives'] + metrics['false_positives']
        total_ai = metrics['false_negatives'] + metrics['true_positives']
        
        print(f"\nText WITHOUT AI mentions (Non-AI Class):")
        print(f"  Total samples: {total_nonai:,}")
        print(f"  ✓ Correctly identified: {metrics['true_negatives']:,} ({metrics['nonai_recall']*100:.2f}%)")
        print(f"  ✗ Incorrectly flagged as AI-related: {metrics['false_positives']:,} ({(metrics['false_positives']/total_nonai)*100:.2f}%)")
        
        print(f"\nText WITH AI mentions (AI Class):")
        print(f"  Total samples: {total_ai:,}")
        print(f"  ✓ Correctly detected: {metrics['true_positives']:,} ({metrics['ai_recall']*100:.2f}%)")
        print(f"  ✗ Missed: {metrics['false_negatives']:,} ({(metrics['false_negatives']/total_ai)*100:.2f}%)")
        
        print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
        print("━" * 80)
        
        # Existing detailed output
        print(f"\n📊 Detailed Performance at threshold={self.threshold:.3f}:")
        print(f"\n   Overall Accuracy: {metrics['accuracy']:.4f}")
        
        print(f"\n   AI Class (Positive):")
        print(f"      Precision: {metrics['ai_precision']:.4f}")
        print(f"      Recall:    {metrics['ai_recall']:.4f}")
        print(f"      F1 Score:  {metrics['ai_f1']:.4f}")
        
        print(f"\n   Non-AI Class (Negative):")
        print(f"      Precision: {metrics['nonai_precision']:.4f}")
        print(f"      Recall:    {metrics['nonai_recall']:.4f}")
        print(f"      F1 Score:  {metrics['nonai_f1']:.4f}")
        
        print(f"\n   Confusion Matrix:")
        print(f"      True Negatives:  {metrics['true_negatives']:>6}")
        print(f"      False Positives: {metrics['false_positives']:>6}")
        print(f"      False Negatives: {metrics['false_negatives']:>6}")
        print(f"      True Positives:  {metrics['true_positives']:>6}")
        
        print(f"\n   Detailed Classification Report:")
        print(class_report)
    
    def save_results(self, results, output_dir=None):
        """Save evaluation results"""
        if output_dir is None:
            output_dir = f"outputs/{self.data_name}/threshold_evaluation"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare results for JSON (remove non-serializable classification report)
        json_results = {}
        for dataset_name, result in results.items():
            json_results[dataset_name] = {
                'metrics': result['metrics'],
                'sample_count': result['sample_count']
            }
        
        # Save complete results as JSON
        results_json = os.path.join(output_dir, f"threshold_eval_{self.threshold}_{timestamp}.json")
        with open(results_json, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        print(f"\n💾 Saved JSON results: {results_json}")
        
        # Save summary CSV
        summary_rows = []
        for dataset_name, result in results.items():
            row = {
                'dataset': dataset_name,
                'sample_count': result['sample_count'],
                **result['metrics']
            }
            summary_rows.append(row)
        
        summary_csv = os.path.join(output_dir, f"threshold_eval_{self.threshold}_{timestamp}.csv")
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"📊 Saved CSV summary: {summary_csv}")
        
        # Save predictions parquet files for error analysis
        print(f"\n💾 Saving prediction parquet files for error analysis...")
        for dataset_name, result in results.items():
            predictions_df = pd.DataFrame({
                'text': result['texts'],
                'true_label': result['true_labels'],
                'predicted_label': result['predicted_labels'],
                'ai_probability': result['ai_probs'],
                'is_correct': result['true_labels'] == result['predicted_labels']
            })
            
            predictions_parquet = os.path.join(
                output_dir, 
                f"predictions_{dataset_name}_{self.threshold}_{timestamp}.parquet"
            )
            predictions_df.to_parquet(predictions_parquet, index=False)
            print(f"   ✓ Saved predictions for {dataset_name}: {predictions_parquet}")
        
        # Save detailed text report
        report_txt = os.path.join(output_dir, f"threshold_eval_{self.threshold}_{timestamp}.txt")
        with open(report_txt, 'w') as f:
            f.write(f"Threshold Evaluation Report\n")
            f.write(f"{'='*80}\n")
            f.write(f"Threshold: {self.threshold:.3f}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"\n\n")
            
            for dataset_name, result in results.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Sample Count: {result['sample_count']}\n\n")
                f.write(result['classification_report'])
                f.write(f"\n\nDetailed Metrics:\n")
                for key, value in result['metrics'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n\n")
        
        print(f"📝 Saved detailed report: {report_txt}")
        
        # Generate confusion matrix plots
        self.plot_confusion_matrices(results, output_dir, timestamp)
    
    def plot_confusion_matrices(self, results, output_dir, timestamp):
        """Plot confusion matrices for each dataset"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        for dataset_name, result in results.items():
            metrics = result['metrics']
            
            # Create confusion matrix
            cm = np.array([
                [metrics['true_negatives'], metrics['false_positives']],
                [metrics['false_negatives'], metrics['true_positives']]
            ])
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Non-AI', 'AI'],
                yticklabels=['Non-AI', 'AI'],
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title(f'Confusion Matrix: {dataset_name}\nThreshold: {self.threshold:.3f}', fontsize=14)
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"confusion_matrix_{dataset_name.replace(' ', '_')}_{timestamp}.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Saved confusion matrix plot: {plot_file}")


def main():
    """Main entry point"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate instruction-tuned model at a single threshold'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/instruction_training_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='outputs/checkpoints/gemma3_20k_balanced_it_v2/final_model',
        help='Path to trained model'
    )
    parser.add_argument(
        '--data-name',
        type=str,
        default='final_data_4_v2',
        help='Dataset name for output organization'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        nargs='+',
        default=[
            './data/final_data_4_v2/ood_distribution_test.parquet',
        ],
        help='Paths to test data parquet files'
    )
    parser.add_argument(
        '--test-names',
        type=str,
        nargs='+',
        default=[
            'ood_distribution_test',
        ],
        help='Names for test datasets'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.test_data) != len(args.test_names):
        parser.error("Number of --test-data and --test-names must match")
    
    print(f"{'='*80}")
    print(f"THRESHOLD EVALUATION")
    print(f"{'='*80}")
    print(f"Threshold: {args.threshold:.3f}")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Test datasets: {len(args.test_data)}")
    
    # Initialize evaluator
    evaluator = ThresholdEvaluator(
        config_path=args.config,
        model_path=args.model,
        data_name=args.data_name,
        threshold=args.threshold
    )
    evaluator.setup_device()
    evaluator.load_model()
    
    # Get predictions (AI probabilities)
    print("\n" + "="*80)
    print("STEP 1: Getting AI Probabilities")
    print("="*80)
    predictions_dict = evaluator.get_predictions(args.test_data, args.test_names)
    
    # Evaluate at threshold
    print("\n" + "="*80)
    print("STEP 2: Evaluating at Threshold")
    print("="*80)
    results = evaluator.evaluate_all_datasets(predictions_dict)
    
    # Save results
    print("\n" + "="*80)
    print("STEP 3: Saving Results")
    print("="*80)
    evaluator.save_results(results)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nEvaluated at threshold: {args.threshold:.3f}")
    print(f"Results saved to: outputs/{args.data_name}/threshold_evaluation/")


if __name__ == "__main__":
    main()

