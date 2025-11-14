#!/usr/bin/env python3
"""
Threshold Optimization Script for Instruction-Tuned Gemma Model
Tests different thresholds to find optimal balance between recall and precision
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
    confusion_matrix, roc_auc_score, precision_recall_curve
)
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


class ThresholdOptimizer:
    """Optimize classification threshold for instruction-tuned model"""
    
    def __init__(self, config_path, model_path, data_name):
        self.config_path = config_path
        self.model_path = model_path
        self.data_name = data_name
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
            dict: {dataset_name: {'true_labels': array, 'ai_probs': array}}
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
    
    def optimize_thresholds(self, predictions_dict, thresholds=None):
        """
        Test multiple thresholds and find optimal ones
        
        Args:
            predictions_dict: Dict from get_predictions()
            thresholds: List of thresholds to test (default: 0.1 to 0.9 in steps of 0.05)
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05)
        
        print(f"\n{'='*80}")
        print(f"THRESHOLD OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Testing {len(thresholds)} thresholds: {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
        
        results = {}
        
        for dataset_name, data in predictions_dict.items():
            print(f"\n\n{'='*80}")
            print(f"Optimizing for: {dataset_name}")
            print(f"{'='*80}")
            
            true_labels = data['true_labels']
            ai_probs = data['ai_probs']
            
            # Test each threshold
            threshold_results = []
            for threshold in tqdm(thresholds, desc="Testing thresholds"):
                metrics = self.evaluate_threshold(true_labels, ai_probs, threshold)
                threshold_results.append(metrics)
            
            # Find optimal thresholds for different goals
            results_df = pd.DataFrame(threshold_results)
            
            # Goal 1: Maximize AI recall (catch all AI content)
            best_recall_idx = results_df['ai_recall'].idxmax()
            best_recall = results_df.iloc[best_recall_idx]
            
            # Goal 2: Maximize F1 score (balance)
            best_f1_idx = results_df['ai_f1'].idxmax()
            best_f1 = results_df.iloc[best_f1_idx]
            
            # Goal 3: Maximize accuracy
            best_acc_idx = results_df['accuracy'].idxmax()
            best_acc = results_df.iloc[best_acc_idx]
            
            # Goal 4: High AI recall (>90%) with best precision
            high_recall_mask = results_df['ai_recall'] >= 0.90
            if high_recall_mask.any():
                high_recall_df = results_df[high_recall_mask]
                best_precision_at_90recall_idx = high_recall_df['ai_precision'].idxmax()
                best_precision_at_90recall = results_df.iloc[best_precision_at_90recall_idx]
            else:
                best_precision_at_90recall = None
            
            results[dataset_name] = {
                'all_thresholds': threshold_results,
                'best_for_recall': best_recall.to_dict(),
                'best_for_f1': best_f1.to_dict(),
                'best_for_accuracy': best_acc.to_dict(),
                'best_precision_at_90recall': best_precision_at_90recall.to_dict() if best_precision_at_90recall is not None else None,
                'current_0.5': self.evaluate_threshold(true_labels, ai_probs, 0.5)
            }
            
            # Print summary
            self.print_optimization_results(dataset_name, results[dataset_name])
        
        return results
    
    def print_optimization_results(self, dataset_name, result):
        """Pretty print optimization results"""
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION RESULTS: {dataset_name}")
        print(f"{'='*80}")
        
        print(f"\n📊 Current Performance (threshold=0.5):")
        curr = result['current_0.5']
        print(f"   AI Recall: {curr['ai_recall']:.4f}")
        print(f"   AI Precision: {curr['ai_precision']:.4f}")
        print(f"   AI F1: {curr['ai_f1']:.4f}")
        print(f"   Accuracy: {curr['accuracy']:.4f}")
        
        print(f"\n🎯 Best for Maximum AI Recall:")
        best_recall = result['best_for_recall']
        print(f"   Threshold: {best_recall['threshold']:.3f}")
        print(f"   AI Recall: {best_recall['ai_recall']:.4f} ⬆️")
        print(f"   AI Precision: {best_recall['ai_precision']:.4f}")
        print(f"   AI F1: {best_recall['ai_f1']:.4f}")
        print(f"   Accuracy: {best_recall['accuracy']:.4f}")
        
        print(f"\n⚖️  Best for Balanced Performance (F1):")
        best_f1 = result['best_for_f1']
        print(f"   Threshold: {best_f1['threshold']:.3f}")
        print(f"   AI Recall: {best_f1['ai_recall']:.4f}")
        print(f"   AI Precision: {best_f1['ai_precision']:.4f}")
        print(f"   AI F1: {best_f1['ai_f1']:.4f} ⬆️")
        print(f"   Accuracy: {best_f1['accuracy']:.4f}")
        
        print(f"\n🎓 Best for Overall Accuracy:")
        best_acc = result['best_for_accuracy']
        print(f"   Threshold: {best_acc['threshold']:.3f}")
        print(f"   AI Recall: {best_acc['ai_recall']:.4f}")
        print(f"   AI Precision: {best_acc['ai_precision']:.4f}")
        print(f"   AI F1: {best_acc['ai_f1']:.4f}")
        print(f"   Accuracy: {best_acc['accuracy']:.4f} ⬆️")
        
        if result['best_precision_at_90recall']:
            print(f"\n🚀 Best Precision at ≥90% AI Recall:")
            best_90 = result['best_precision_at_90recall']
            print(f"   Threshold: {best_90['threshold']:.3f}")
            print(f"   AI Recall: {best_90['ai_recall']:.4f} (≥90%)")
            print(f"   AI Precision: {best_90['ai_precision']:.4f} ⬆️")
            print(f"   AI F1: {best_90['ai_f1']:.4f}")
            print(f"   Accuracy: {best_90['accuracy']:.4f}")
    
    def save_results(self, results, output_dir=None):
        """Save optimization results"""
        if output_dir is None:
            output_dir = f"outputs/{self.data_name}/threshold_optimization"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results as JSON
        results_json = os.path.join(output_dir, f"threshold_optimization_{timestamp}.json")
        with open(results_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Saved: {results_json}")
        
        # Save summary CSV
        summary_rows = []
        for dataset_name, result in results.items():
            for goal_name, goal_key in [
                ('Current (0.5)', 'current_0.5'),
                ('Best Recall', 'best_for_recall'),
                ('Best F1', 'best_for_f1'),
                ('Best Accuracy', 'best_for_accuracy'),
            ]:
                if goal_key in result and result[goal_key]:
                    row = {
                        'dataset': dataset_name,
                        'optimization_goal': goal_name,
                        **result[goal_key]
                    }
                    summary_rows.append(row)
            
            if result.get('best_precision_at_90recall'):
                row = {
                    'dataset': dataset_name,
                    'optimization_goal': 'Best Precision at ≥90% Recall',
                    **result['best_precision_at_90recall']
                }
                summary_rows.append(row)
        
        summary_csv = os.path.join(output_dir, f"threshold_summary_{timestamp}.csv")
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"📊 Saved: {summary_csv}")
        
        # Generate plots
        self.plot_threshold_curves(results, output_dir, timestamp)
    
    def plot_threshold_curves(self, results, output_dir, timestamp):
        """Plot threshold vs metrics curves"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        for dataset_name, result in results.items():
            df = pd.DataFrame(result['all_thresholds'])
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Threshold Optimization: {dataset_name}', fontsize=16)
            
            # Plot 1: Recall vs Threshold
            axes[0, 0].plot(df['threshold'], df['ai_recall'], 'b-', label='AI Recall', linewidth=2)
            axes[0, 0].plot(df['threshold'], df['nonai_recall'], 'r--', label='Non-AI Recall', linewidth=2)
            axes[0, 0].axvline(x=0.5, color='gray', linestyle=':', label='Current (0.5)')
            axes[0, 0].set_xlabel('Threshold')
            axes[0, 0].set_ylabel('Recall')
            axes[0, 0].set_title('Recall vs Threshold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Precision vs Threshold
            axes[0, 1].plot(df['threshold'], df['ai_precision'], 'b-', label='AI Precision', linewidth=2)
            axes[0, 1].plot(df['threshold'], df['nonai_precision'], 'r--', label='Non-AI Precision', linewidth=2)
            axes[0, 1].axvline(x=0.5, color='gray', linestyle=':', label='Current (0.5)')
            axes[0, 1].set_xlabel('Threshold')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision vs Threshold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: F1 vs Threshold
            axes[1, 0].plot(df['threshold'], df['ai_f1'], 'g-', label='AI F1', linewidth=2)
            axes[1, 0].plot(df['threshold'], df['accuracy'], 'purple', label='Accuracy', linewidth=2)
            axes[1, 0].axvline(x=0.5, color='gray', linestyle=':', label='Current (0.5)')
            axes[1, 0].set_xlabel('Threshold')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('F1 & Accuracy vs Threshold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Precision-Recall Curve
            axes[1, 1].plot(df['ai_recall'], df['ai_precision'], 'b-', linewidth=2)
            axes[1, 1].set_xlabel('Recall')
            axes[1, 1].set_ylabel('Precision')
            axes[1, 1].set_title('Precision-Recall Curve')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Mark current threshold
            curr = result['current_0.5']
            axes[1, 1].plot(curr['ai_recall'], curr['ai_precision'], 'ro', markersize=10, label='Current (0.5)')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_{timestamp}.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Saved plot: {plot_file}")


def main():
    """Main entry point"""
    # Configuration
    config_path = "config/instruction_training_config.yaml"
    model_path = "outputs/checkpoints/gemma3_20k_balanced_it_v2/final_model"
    data_name = "final_data_4_v2"
    # Define test datasets
    parquet_paths = [
        './data/final_data_4_v2/iid_distribution_test.parquet',
        './data/final_data_4_v2/ood_distribution_test.parquet',
    ]
    
    dataset_names = [
        'iid_distribution_test',
        'ood_distribution_test',
    ]
    
    # Initialize optimizer
    optimizer = ThresholdOptimizer(config_path, model_path, data_name)
    optimizer.setup_device()
    optimizer.load_model()
    
    # Get predictions (AI probabilities)
    print("\n" + "="*80)
    print("STEP 1: Getting AI Probabilities")
    print("="*80)
    predictions_dict = optimizer.get_predictions(parquet_paths, dataset_names)
    
    # Optimize thresholds
    print("\n" + "="*80)
    print("STEP 2: Optimizing Thresholds")
    print("="*80)
    results = optimizer.optimize_thresholds(predictions_dict)
    
    # Save results
    print("\n" + "="*80)
    print("STEP 3: Saving Results")
    print("="*80)
    optimizer.save_results(results)
    
    print("\n" + "="*80)
    print("✅ THRESHOLD OPTIMIZATION COMPLETE!")
    print("="*80)
    print("\nRecommendations:")
    print("1. Review the plots in outputs/threshold_optimization/")
    print("2. Check the CSV summary for quick comparison")
    print("3. Choose threshold based on your priority:")
    print("   - High recall: Catch more AI content (lower false negatives)")
    print("   - High precision: Fewer false alarms (lower false positives)")
    print("   - Balanced: Best F1 score")


if __name__ == "__main__":
    main()