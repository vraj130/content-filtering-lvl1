#!/usr/bin/env python3
"""
Threshold Evaluation Script for FastText Classifier
Evaluates FastText model across multiple thresholds (0.1-0.9) on multiple datasets.

Usage:
    python scripts/evaluate_fasttext.py \
        --model models/fasttext/fasttext_model.bin \
        --test data/eval_all/text_format/alignment_dataset_ai_positive_v3_test.txt \
               data/eval_all/text_format/fineweb_ai_negative_v3_test.txt \
               data/eval_all/text_format/fineweb_ai_positive_v3_test.txt
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.fasttext_inference import FastTextClassifier

# Default thresholds to evaluate
DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_test_data(test_file: str) -> Tuple[List[str], List[int]]:
    """
    Load test data from fastText format file.
    
    Args:
        test_file: Path to test file in fastText format
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('__label__ai'):
                labels.append(1)
                texts.append(line[len('__label__ai'):].strip())
            elif line.startswith('__label__nonai'):
                labels.append(0)
                texts.append(line[len('__label__nonai'):].strip())
    
    return texts, labels


def predict_and_cache(classifier: FastTextClassifier, test_files: List[str]) -> pd.DataFrame:
    """
    Run predictions once on all datasets and cache results.
    
    Args:
        classifier: Loaded FastTextClassifier
        test_files: List of test file paths
        
    Returns:
        DataFrame with columns: text, true_label, ai_probability, dataset_name
    """
    print("\n" + "="*80)
    print("PREDICTION CACHING PHASE")
    print("="*80)
    
    all_results = []
    
    for test_file in test_files:
        dataset_name = Path(test_file).name
        print(f"\n📂 Processing: {dataset_name}")
        
        # Load data
        texts, labels = load_test_data(test_file)
        print(f"   Loaded {len(texts):,} samples")
        print(f"   AI samples: {sum(labels):,}, Non-AI samples: {len(labels) - sum(labels):,}")
        
        # Get predictions with progress bar
        for i, (text, label) in enumerate(tqdm(zip(texts, labels), total=len(texts), desc=f"   Predicting")):
            scores = classifier.predict_with_all_scores(text)
            ai_prob = scores['ai']
            
            all_results.append({
                'text': text,
                'true_label': label,
                'ai_probability': ai_prob,
                'dataset_name': dataset_name
            })
    
    cached_df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("CACHING COMPLETE")
    print("="*80)
    print(f"Total samples cached: {len(cached_df):,}")
    print(f"Datasets: {cached_df['dataset_name'].nunique()}")
    
    return cached_df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    """
    Calculate comprehensive metrics for a given threshold.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: AI probabilities
        
    Returns:
        Dictionary of metrics
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    
    # Confidence statistics
    correct_mask = y_true == y_pred
    
    if correct_mask.sum() > 0:
        mean_confidence_correct = np.mean([
            y_prob[i] if y_pred[i] == 1 else (1 - y_prob[i])
            for i in range(len(y_pred)) if correct_mask[i]
        ])
    else:
        mean_confidence_correct = 0.0
    
    if (~correct_mask).sum() > 0:
        mean_confidence_incorrect = np.mean([
            y_prob[i] if y_pred[i] == 1 else (1 - y_prob[i])
            for i in range(len(y_pred)) if not correct_mask[i]
        ])
    else:
        mean_confidence_incorrect = 0.0
    
    mean_confidence_overall = np.mean([
        y_prob[i] if y_pred[i] == 1 else (1 - y_prob[i])
        for i in range(len(y_pred))
    ])
    
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


def evaluate_thresholds(cached_df: pd.DataFrame, thresholds: List[float]) -> Tuple[Dict, float, Dict]:
    """
    Evaluate performance across multiple thresholds.
    
    Args:
        cached_df: DataFrame with cached predictions
        thresholds: List of thresholds to evaluate
        
    Returns:
        Tuple of (results_dict, roc_auc_combined, per_dataset_roc_auc)
    """
    print("\n" + "="*80)
    print("THRESHOLD EVALUATION PHASE")
    print("="*80)
    
    results = {}
    
    # Calculate ROC-AUC once (threshold-independent)
    y_true_combined = cached_df['true_label'].values
    y_prob_combined = cached_df['ai_probability'].values
    
    try:
        roc_auc_combined = float(roc_auc_score(y_true_combined, y_prob_combined))
        print(f"\n📊 Combined ROC-AUC: {roc_auc_combined:.4f}")
    except Exception as e:
        print(f"\n⚠️  Could not calculate combined ROC-AUC: {e}")
        roc_auc_combined = None
    
    # Calculate per-dataset ROC-AUC
    per_dataset_roc_auc = {}
    for dataset_name in cached_df['dataset_name'].unique():
        dataset_df = cached_df[cached_df['dataset_name'] == dataset_name]
        try:
            roc_auc = roc_auc_score(
                dataset_df['true_label'].values,
                dataset_df['ai_probability'].values
            )
            per_dataset_roc_auc[dataset_name] = float(roc_auc)
            print(f"   {dataset_name}: ROC-AUC = {roc_auc:.4f}")
        except Exception as e:
            per_dataset_roc_auc[dataset_name] = None
            print(f"   {dataset_name}: ROC-AUC = N/A ({e})")
    
    # Evaluate each threshold
    print(f"\n📏 Evaluating {len(thresholds)} thresholds...")
    
    for threshold in tqdm(thresholds, desc="Thresholds"):
        # Apply threshold to get predictions
        cached_df['predicted_label'] = (cached_df['ai_probability'] >= threshold).astype(int)
        
        # Combined metrics
        y_true = cached_df['true_label'].values
        y_pred = cached_df['predicted_label'].values
        y_prob = cached_df['ai_probability'].values
        
        combined_metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        # Per-dataset metrics
        per_dataset_metrics = {}
        for dataset_name in cached_df['dataset_name'].unique():
            dataset_df = cached_df[cached_df['dataset_name'] == dataset_name]
            
            y_true_ds = dataset_df['true_label'].values
            y_pred_ds = dataset_df['predicted_label'].values
            y_prob_ds = dataset_df['ai_probability'].values
            
            dataset_metrics = calculate_metrics(y_true_ds, y_pred_ds, y_prob_ds)
            per_dataset_metrics[dataset_name] = dataset_metrics
        
        # Store results
        results[str(threshold)] = {
            'combined': combined_metrics,
            'per_dataset': per_dataset_metrics
        }
    
    # Print per-dataset summary tables
    dataset_names = cached_df['dataset_name'].unique()
    
    for dataset_name in dataset_names:
        print("\n" + "="*80)
        print(f"THRESHOLD SUMMARY: {dataset_name}, total samples: {len(cached_df[cached_df['dataset_name'] == dataset_name])}")
        print("="*80)
        
        # Get ROC-AUC for this dataset
        ds_roc_auc = per_dataset_roc_auc.get(dataset_name)
        if ds_roc_auc is not None:
            print(f"ROC-AUC: {ds_roc_auc:.4f}")
        else:
            print("ROC-AUC: N/A (single-class dataset)")
        
        print(f"\n{'Threshold':<12} {'Accuracy':>10} {'AI Recall':>12} {'AI Prec':>10} {'AI F1':>10} {'NonAI Recall':>14} {'NonAI Prec':>12}")
        print("-" * 95)
        
        for threshold in thresholds:
            m = results[str(threshold)]['per_dataset'].get(dataset_name, {})
            if m:
                print(f"{threshold:<12.1f} {m['accuracy']:>10.4f} {m['ai_recall']:>12.4f} {m['ai_precision']:>10.4f} {m['ai_f1']:>10.4f} {m['nonai_recall']:>14.4f} {m['nonai_precision']:>12.4f}")
    
    # Print combined summary table
    print("\n" + "="*80)
    print("THRESHOLD EVALUATION SUMMARY (COMBINED)")
    print("="*80)
    if roc_auc_combined is not None:
        print(f"Combined ROC-AUC: {roc_auc_combined:.4f}")
    print(f"\n{'Threshold':<12} {'Accuracy':>10} {'AI Recall':>12} {'AI Precision':>14} {'AI F1':>10}")
    print("-" * 60)
    
    for threshold in thresholds:
        m = results[str(threshold)]['combined']
        print(f"{threshold:<12.1f} {m['accuracy']:>10.4f} {m['ai_recall']:>12.4f} {m['ai_precision']:>14.4f} {m['ai_f1']:>10.4f}")
    
    return results, roc_auc_combined, per_dataset_roc_auc


def save_results_to_json(
    model_path: str,
    datasets: List[str],
    thresholds: List[float],
    results: Dict,
    roc_auc_combined: float,
    per_dataset_roc_auc: Dict,
    output_dir: str
) -> str:
    """
    Save evaluation results to JSON file.
    
    Args:
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
    output_file = os.path.join(output_dir, f"fasttext_threshold_eval_{timestamp}.json")
    
    output_data = {
        'model_type': 'fasttext',
        'model_path': model_path,
        'datasets': datasets,
        'thresholds': thresholds,
        'roc_auc_combined': roc_auc_combined,
        'per_dataset_roc_auc': per_dataset_roc_auc,
        'results': results,
        'evaluation_timestamp': timestamp
    }
    
    # Custom JSON encoder to handle NaN values (convert to null)
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    
    output_data_clean = clean_for_json(output_data)
    
    with open(output_file, 'w') as f:
        json.dump(output_data_clean, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate FastText classifier across multiple thresholds'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/fasttext/fasttext_model.bin',
        help='Path to trained FastText model'
    )
    parser.add_argument(
        '--test',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to test data in fastText format'
    )
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=DEFAULT_THRESHOLDS,
        help=f'Thresholds to evaluate (default: {DEFAULT_THRESHOLDS})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/fasttext',
        help='Directory to save evaluation results'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("FASTTEXT THRESHOLD EVALUATION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Datasets: {len(args.test)}")
    for f in args.test:
        print(f"  - {f}")
    print("="*80)
    
    # Validate files exist
    for test_file in args.test:
        if not os.path.exists(test_file):
            print(f"❌ Error: File not found: {test_file}")
            sys.exit(1)
    
    # Load model
    print("\n📥 Loading model...")
    classifier = FastTextClassifier(args.model)
    print("✅ Model loaded")
    
    # Predict and cache (run predictions only once)
    cached_df = predict_and_cache(classifier, args.test)
    
    # Evaluate across all thresholds
    results, roc_auc_combined, per_dataset_roc_auc = evaluate_thresholds(
        cached_df, args.thresholds
    )
    
    # Save results
    dataset_names = [Path(f).name for f in args.test]
    output_file = save_results_to_json(
        model_path=args.model,
        datasets=dataset_names,
        thresholds=args.thresholds,
        results=results,
        roc_auc_combined=roc_auc_combined,
        per_dataset_roc_auc=per_dataset_roc_auc,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
