#!/usr/bin/env python3
"""
Parquet Testing Script for ModernBERT Classification Model
Processes parquet files using logits-based classification predictions
"""

import os
import sys
import yaml
import torch
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


class ParquetTester:
    """Process parquet files with ModernBERT classification predictions"""
    
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
        """Load trained ModernBERT classification model and tokenizer"""
        print(f"\n🤖 Loading model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side='right'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Tokenizer loaded")
        
        # Load classification model
        print(f"🔧 Loading ModernBERT classification model...")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.config['training']['bf16'] else torch.float32,
            device_map="auto"
        )
        
        self.model.eval()
        
        print("✅ Model loaded successfully")
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Total parameters: {total_params:,}")
        print(f"   Number of labels: {self.model.config.num_labels}")
        print(f"   Label mapping: {self.model.config.id2label}\n")
    
    def predict_single(self, text, threshold=0.5):
        """
        Predict class for a single text using classification head
        
        Args:
            text: Input text
            threshold: Classification threshold for AI probability
        
        Returns:
            predicted_class (0 or 1), ai_probability (float), nonai_probability (float),
            confidence (float), ai_logit (float), nonai_logit (float)
        """
        # Tokenize
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
        
        # Extract logits for both classes
        nonai_logit = logits[0].item()  # Class 0 = non-AI
        ai_logit = logits[1].item()      # Class 1 = AI
        
        # Calculate probabilities
        probs = F.softmax(logits, dim=0)
        nonai_probability = probs[0].item()
        ai_probability = probs[1].item()
        
        # Predict class using threshold on probability
        predicted_class = 1 if ai_probability >= threshold else 0
        confidence = probs[predicted_class].item()
        
        return predicted_class, ai_probability, nonai_probability, confidence, ai_logit, nonai_logit
    
    def process_parquet(self, parquet_path, threshold=0.5):
        """
        Process all samples in a parquet file
        
        Args:
            parquet_path: Path to input parquet file
            threshold: Classification threshold for AI probability (default: 0.5)
        
        Returns:
            DataFrame with predictions
        """
        print("="*80)
        print("PARQUET PROCESSING MODE - ModernBERT Classification")
        print("="*80)
        print(f"Input file: {parquet_path}")
        print(f"Classification threshold: {threshold}")
        print("="*80)
        
        # Load parquet file
        df = pd.read_parquet(parquet_path)
        print(f"\n✅ Loaded {len(df):,} samples from parquet file")
        
        # Check if labels exist
        has_labels = 'label' in df.columns
        if has_labels:
            print(f"📊 Dataset has labels - accuracy will be calculated")
        else:
            print(f"⚠️  No labels found - predictions only")
        
        # Process each sample
        results = []
        correct_count = 0
        total_with_labels = 0
        
        print(f"\n🔄 Processing samples...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            text = row['text']
            true_label = row.get('label', None)
            
            # Run prediction
            pred_class, ai_prob, nonai_prob, confidence, ai_logit, nonai_logit = self.predict_single(text, threshold=threshold)
            
            # Calculate if correct
            is_correct = None
            if true_label is not None:
                total_with_labels += 1
                is_correct = (pred_class == true_label)
                if is_correct:
                    correct_count += 1
            
            # Store result
            result = {
                'text': text,
                'true_label': true_label,
                'predicted_label': pred_class,
                'ai_probability': ai_prob,
                'nonai_probability': nonai_prob,
                'confidence': confidence,
                'ai_logit': ai_logit,
                'nonai_logit': nonai_logit,
                'is_correct': is_correct,
            }
            results.append(result)
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Print summary
        print("\n" + "="*80)
        print("PROCESSING COMPLETE - SUMMARY")
        print("="*80)
        print(f"Total samples processed: {len(results_df):,}")
        
        if total_with_labels > 0:
            accuracy = correct_count / total_with_labels
            print(f"Samples with labels: {total_with_labels:,}")
            print(f"Correct predictions: {correct_count:,}/{total_with_labels:,} ({accuracy*100:.2f}%)")
            
            # Confusion Matrix
            true_labels = results_df[results_df['true_label'].notna()]['true_label'].astype(int).tolist()
            pred_labels = results_df[results_df['true_label'].notna()]['predicted_label'].astype(int).tolist()
            cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
            
            print(f"\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"                Non-AI    AI")
            print(f"Actual Non-AI   {cm[0,0]:<9} {cm[0,1]}")
            print(f"Actual AI       {cm[1,0]:<9} {cm[1,1]}")
        
        # Count predictions
        ai_count = (results_df['predicted_label'] == 1).sum()
        nonai_count = len(results_df) - ai_count
        print(f"\nPrediction distribution:")
        print(f"  Predicted AI: {ai_count:,}/{len(results_df):,} ({ai_count/len(results_df)*100:.1f}%)")
        print(f"  Predicted Non-AI: {nonai_count:,}/{len(results_df):,} ({nonai_count/len(results_df)*100:.1f}%)")
        
        # Average metrics
        avg_ai_prob = results_df['ai_probability'].mean()
        avg_nonai_prob = results_df['nonai_probability'].mean()
        avg_confidence = results_df['confidence'].mean()
        print(f"\nAverage AI probability: {avg_ai_prob:.4f}")
        print(f"Average Non-AI probability: {avg_nonai_prob:.4f}")
        print(f"Average confidence: {avg_confidence:.4f}")
        
        # Logit statistics
        avg_ai_logit = results_df['ai_logit'].mean()
        avg_nonai_logit = results_df['nonai_logit'].mean()
        print(f"\nAverage AI logit: {avg_ai_logit:.4f}")
        print(f"Average Non-AI logit: {avg_nonai_logit:.4f}")
        
        print("="*80)
        
        return results_df
    
    def save_predictions(self, results_df, output_path=None, parquet_name="predictions"):
        """
        Save predictions to parquet file
        
        Args:
            results_df: DataFrame with predictions
            output_path: Output directory (default: outputs/parquet_predictions)
            parquet_name: Base name for output file
        """
        if output_path is None:
            output_path = "outputs/parquet_predictions"
        
        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions parquet
        output_file = os.path.join(output_path, f"{parquet_name}_{timestamp}.parquet")
        results_df.to_parquet(output_file, index=False)
        print(f"\n💾 Saved predictions to: {output_file}")
        
        # Also save CSV for easy viewing
        csv_file = os.path.join(output_path, f"{parquet_name}_{timestamp}.csv")
        results_df.to_csv(csv_file, index=False)
        print(f"📊 Saved CSV to: {csv_file}")
        
        return output_file


def main():
    """Main entry point"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process parquet file with ModernBERT classification predictions'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input parquet file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/modernbert_training_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='outputs/checkpoints/modernbert_large_1114/final_model',
        help='Path to trained model'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/parquet_predictions',
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='modernbert_predictions',
        help='Base name for output files'
    )
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f"MODERNBERT PARQUET PREDICTION PROCESSING")
    print(f"{'='*80}")
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output}")
    print(f"{'='*80}\n")
    
    # Initialize tester
    tester = ParquetTester(args.config, args.model)
    tester.setup_device()
    tester.load_model()
    
    # Process parquet file
    results_df = tester.process_parquet(
        args.input, 
        threshold=args.threshold
    )
    
    # Save predictions
    tester.save_predictions(results_df, args.output, args.name)
    
    print("\n✅ Processing complete!")


if __name__ == "__main__":
    main()

