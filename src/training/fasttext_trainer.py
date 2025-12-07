"""
FastText Trainer for Binary Classification (AI vs non-AI content)
Following existing trainer patterns in the codebase.
"""

import os
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, accuracy_score
)


def safe_predict(model, text: str, k: int = 1):
    """
    Safely call fasttext predict, handling NumPy 2.0 compatibility issues.
    
    Args:
        model: fastText model
        text: Input text
        k: Number of predictions to return
        
    Returns:
        Tuple of (labels, probabilities)
    """
    try:
        labels, probs = model.predict(text, k=k)
        return labels, probs
    except ValueError as e:
        if "copy" in str(e):
            # NumPy 2.0 compatibility workaround
            # Call predict with k=1 multiple times and aggregate
            result_labels = []
            result_probs = []
            for i in range(k):
                try:
                    # Try with single prediction
                    label, prob = model.predict(text, k=1)
                    result_labels.extend(label)
                    result_probs.extend([float(prob[0])] if hasattr(prob, '__iter__') else [float(prob)])
                except:
                    break
            if result_labels:
                return tuple(result_labels), np.array(result_probs)
            # Fallback: return empty
            return ('__label__nonai',), np.array([0.5])
        raise


class FastTextTrainer:
    """Training pipeline for fastText binary classifier."""
    
    def __init__(self, config: dict):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config
        self.model = None
        self.logger = None
        self.class2id = config.get('class_mapping', {'nonai': 0, 'ai': 1})
        self.id2class = {v: k for k, v in self.class2id.items()}
        
    def setup_logging(self):
        """Setup comprehensive logging to both console and file."""
        log_dir = self.config.get('output', {}).get('log_dir', 'outputs/logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"fasttext_training_{timestamp}.log")
        
        self.logger = logging.getLogger('FastTextTrainer')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info("="*80)
        self.logger.info("FastText Trainer - Logging Initialized")
        self.logger.info("="*80)
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("-"*80)
        
    def validate_data_files(self) -> bool:
        """Validate that required data files exist."""
        data_config = self.config.get('data', {})
        data_dir = data_config.get('data_dir', 'data/content_filtering_dataset_fasttext')
        
        train_file = os.path.join(data_dir, data_config.get('train_file', 'train.txt'))
        val_file = os.path.join(data_dir, data_config.get('val_file', 'val.txt'))
        
        if not os.path.exists(train_file):
            self.logger.error(f"Training file not found: {train_file}")
            return False
            
        if not os.path.exists(val_file):
            self.logger.warning(f"Validation file not found: {val_file}")
            # Continue without validation
            
        self.logger.info(f"Training file: {train_file}")
        self.logger.info(f"Validation file: {val_file}")
        
        # Count samples
        with open(train_file, 'r', encoding='utf-8') as f:
            train_count = sum(1 for _ in f)
        self.logger.info(f"Training samples: {train_count:,}")
        
        if os.path.exists(val_file):
            with open(val_file, 'r', encoding='utf-8') as f:
                val_count = sum(1 for _ in f)
            self.logger.info(f"Validation samples: {val_count:,}")
        
        return True
    
    def train(self) -> fasttext.FastText:
        """
        Train fastText model with configured hyperparameters.
        
        Returns:
            Trained fastText model
        """
        data_config = self.config.get('data', {})
        training_config = self.config.get('training', {})
        
        data_dir = data_config.get('data_dir', 'data/content_filtering_dataset_fasttext')
        train_file = os.path.join(data_dir, data_config.get('train_file', 'train.txt'))
        
        # Hyperparameters
        epochs = training_config.get('epochs', 25)
        lr = training_config.get('lr', 0.5)
        word_ngrams = training_config.get('wordNgrams', 2)
        dim = training_config.get('dim', 100)
        loss = training_config.get('loss', 'softmax')
        min_count = training_config.get('minCount', 1)
        bucket = training_config.get('bucket', 2000000)
        
        self.logger.info("="*80)
        self.logger.info("Starting FastText Training")
        self.logger.info("="*80)
        self.logger.info(f"Training file: {train_file}")
        self.logger.info(f"Hyperparameters:")
        self.logger.info(f"  epochs: {epochs}")
        self.logger.info(f"  learning_rate: {lr}")
        self.logger.info(f"  wordNgrams: {word_ngrams}")
        self.logger.info(f"  dim: {dim}")
        self.logger.info(f"  loss: {loss}")
        self.logger.info(f"  minCount: {min_count}")
        self.logger.info("-"*80)
        
        start_time = time.time()
        
        # Train model
        self.model = fasttext.train_supervised(
            input=train_file,
            epoch=epochs,
            lr=lr,
            wordNgrams=word_ngrams,
            dim=dim,
            loss=loss,
            minCount=min_count,
            bucket=bucket,
            verbose=2
        )
        
        training_time = time.time() - start_time
        self.logger.info(f"\n✅ Training completed in {training_time:.2f} seconds")
        
        return self.model
    
    def evaluate(self, split: str = 'val') -> Dict:
        """
        Evaluate model on specified split.
        
        Args:
            split: 'val' or 'test'
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        data_config = self.config.get('data', {})
        data_dir = data_config.get('data_dir', 'data/content_filtering_dataset_fasttext')
        eval_file = os.path.join(data_dir, data_config.get(f'{split}_file', f'{split}.txt'))
        

        # DEBUG: Test if predict works properly
        self.logger.info("\n=== DEBUG: Testing model.predict ===")
        try:
            test_labels, test_probs = self.model.predict("This is a test sentence", k=2)
            self.logger.info(f"Labels: {test_labels}")
            self.logger.info(f"Probs: {test_probs}")
            self.logger.info(f"Probs type: {type(test_probs)}")
        except Exception as e:
            self.logger.error(f"Direct predict failed: {e}")
        
        try:
            test_labels2, test_probs2 = safe_predict(self.model, "This is a test sentence", k=2)
            self.logger.info(f"Safe predict - Labels: {test_labels2}")
            self.logger.info(f"Safe predict - Probs: {test_probs2}")
        except Exception as e:
            self.logger.error(f"Safe predict failed: {e}")
        self.logger.info("=== END DEBUG ===\n")


        
        if not os.path.exists(eval_file):
            self.logger.warning(f"Evaluation file not found: {eval_file}")
            return {}
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Evaluating on {split} set: {eval_file}")
        self.logger.info(f"{'='*80}")
        
        # FastText built-in evaluation
        n_samples, precision, recall = self.model.test(eval_file)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.logger.info(f"FastText Metrics (Overall):")
        self.logger.info(f"  Samples: {n_samples:,}")
        self.logger.info(f"  Precision: {precision:.4f}")
        self.logger.info(f"  Recall: {recall:.4f}")
        self.logger.info(f"  F1: {f1:.4f}")
        
        # Detailed per-class evaluation
        true_labels = []
        pred_labels = []
        pred_probs = []
        
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Extract true label
                if line.startswith('__label__ai'):
                    true_labels.append(1)
                    text = line[len('__label__ai'):].strip()
                elif line.startswith('__label__nonai'):
                    true_labels.append(0)
                    text = line[len('__label__nonai'):].strip()
                else:
                    continue
                
                # Get prediction (using safe wrapper for NumPy 2.0 compatibility)
                labels, probs = safe_predict(self.model, text, k=2)
                
                # Extract AI probability explicitly by finding the AI label
                ai_prob = 0.5  # default fallback
                for i, label in enumerate(labels):
                    if label == '__label__ai':
                        ai_prob = float(probs[i])
                        break

                pred_labels.append(1 if ai_prob >= 0.5 else 0)
                pred_probs.append(ai_prob)      
        
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        pred_probs = np.array(pred_probs)
        
        # Calculate detailed metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
            true_labels, pred_labels, labels=[0, 1], zero_division=0
        )
        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
        
        metrics = {
            'accuracy': accuracy,
            'overall_precision': precision,
            'overall_recall': recall,
            'overall_f1': f1,
            'nonai_precision': precision_arr[0],
            'nonai_recall': recall_arr[0],
            'nonai_f1': f1_arr[0],
            'ai_precision': precision_arr[1],
            'ai_recall': recall_arr[1],
            'ai_f1': f1_arr[1],
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1]),
            'n_samples': n_samples
        }
        
        # Log detailed metrics
        self.logger.info(f"\nDetailed Metrics:")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"\n  Non-AI Class:")
        self.logger.info(f"    Precision: {metrics['nonai_precision']:.4f}")
        self.logger.info(f"    Recall: {metrics['nonai_recall']:.4f}")
        self.logger.info(f"    F1: {metrics['nonai_f1']:.4f}")
        self.logger.info(f"\n  AI Class:")
        self.logger.info(f"    Precision: {metrics['ai_precision']:.4f}")
        self.logger.info(f"    Recall: {metrics['ai_recall']:.4f}")
        self.logger.info(f"    F1: {metrics['ai_f1']:.4f}")
        
        self.logger.info(f"\nConfusion Matrix:")
        self.logger.info(f"                Predicted")
        self.logger.info(f"                non-AI    AI")
        self.logger.info(f"Actual non-AI   {cm[0,0]:<9} {cm[0,1]}")
        self.logger.info(f"Actual AI       {cm[1,0]:<9} {cm[1,1]}")
        
        return metrics
    
    def save_model(self) -> str:
        """
        Save trained model to disk.
        
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        output_config = self.config.get('output', {})
        model_dir = output_config.get('model_dir', 'models/fasttext')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'fasttext_model.bin')
        self.model.save_model(model_path)
        
        self.logger.info(f"\n💾 Model saved to: {model_path}")
        
        return model_path
    
    def save_metrics(self, metrics: Dict, split: str = 'val'):
        """Save evaluation metrics to JSON file."""
        output_config = self.config.get('output', {})
        output_dir = output_config.get('metrics_dir', 'outputs/fasttext')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(output_dir, f"metrics_{split}_{timestamp}.json")
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"📊 Metrics saved to: {metrics_path}")
    
    def quick_test(self, test_texts: list = None):
        """Quick test on sample texts."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if test_texts is None:
            test_texts = [
                "We discuss reinforcement learning from human feedback and its applications in AI safety.",
                "The recipe for chocolate chip cookies requires butter, sugar, and flour.",
                "Constitutional AI aims to align language models with human values through iterative refinement.",
                "I went to the park yesterday and saw many dogs playing."
            ]
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Quick Test Predictions")
        self.logger.info(f"{'='*80}")
        
        for text in test_texts:
            labels, probs = safe_predict(self.model, text, k=2)
            pred_label = labels[0].replace('__label__', '')
            confidence = float(probs[0])
            
            self.logger.info(f"\nText: {text[:60]}...")
            self.logger.info(f"Prediction: {pred_label} (confidence: {confidence:.4f})")
    
    def run_full_pipeline(self):
        """Execute the complete training pipeline."""
        # Setup
        self.setup_logging()
        
        self.logger.info("="*80)
        self.logger.info("FastText Binary Classifier Training Pipeline")
        self.logger.info("="*80)
        self.logger.info(f"Task: Binary Classification (AI vs non-AI content)")
        self.logger.info("-"*80)
        
        # Validate data
        if not self.validate_data_files():
            self.logger.error("Data validation failed. Aborting.")
            return
        
        # Train
        self.train()
        
        # Evaluate on validation set
        val_metrics = self.evaluate('val')
        if val_metrics:
            self.save_metrics(val_metrics, 'val')
        
        # Evaluate on test set if available
        test_metrics = self.evaluate('test')
        if test_metrics:
            self.save_metrics(test_metrics, 'test')
        
        # Save model
        model_path = self.save_model()
        
        # Quick test
        self.quick_test()
        
        # Summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info("✅ Training Pipeline Complete!")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Model saved to: {model_path}")
        if val_metrics:
            self.logger.info(f"Validation Accuracy: {val_metrics.get('accuracy', 0):.4f}")
            self.logger.info(f"Validation AI Recall: {val_metrics.get('ai_recall', 0):.4f}")
            self.logger.info(f"Validation AI F1: {val_metrics.get('ai_f1', 0):.4f}")
        if test_metrics:
            self.logger.info(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            self.logger.info(f"Test AI Recall: {test_metrics.get('ai_recall', 0):.4f}")
            self.logger.info(f"Test AI F1: {test_metrics.get('ai_f1', 0):.4f}")
        self.logger.info("="*80)

