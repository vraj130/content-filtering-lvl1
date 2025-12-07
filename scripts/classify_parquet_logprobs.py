#!/usr/bin/env python3
"""
Parquet API Classification Script with Multi-Threshold Evaluation

Iterates through multiple parquet files, fills a template with each row's text,
sends it to an OpenAI-compatible API (vLLM), and evaluates yes/no responses
against the label column using multiple probability thresholds.

Can be run for multiple dataset at once like this - 

python ./scripts/classify_parquet_logprobs.py \
    --input data/eval_all/parquet_format/alignment_dataset_ai_positive_v3_test.parquet \
        data/eval_all/parquet_format/fineweb_ai_negative_v3_test.parquet \
        data/eval_all/parquet_format/fineweb_ai_positive_v3_test.parquet \
    --template config/instruction_training_config.yaml \
    --api-base http://localhost:8000/v1 \
    --api-key placeholder \
    --model stage-1-classifier \

"""




import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process parquet files through OpenAI-compatible API and evaluate responses across multiple thresholds"
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to input parquet file(s)",
    )
    parser.add_argument(
        "--template",
        type=str,
        required=False,
        help="Path to template file with {text} placeholder",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL (overrides OPENAI_API_BASE env var)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="placeholder",
        help="API key (overrides OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stage-1-classifier",
        help="Model name to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: results_<timestamp>.json)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the text column in parquet (default: text)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="Maximum tokens for API response (default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for API response (default: 0.0)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Classification thresholds for 'yes' probability (default: 0.1-0.9)",
    )
    return parser.parse_args()


def load_config(config_path: str):
    """Load configuration from YAML"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_template(template_path: str) -> str:
    """Load template from file or config."""
    config = load_config("config/instruction_training_config.yaml")
    template = config["instruction"]["prompt_template"]

    if "{text}" not in template:
        print("WARNING: Template does not contain {text} placeholder")

    return template


def normalize_label(label) -> int:
    """Normalize label to 0 or 1 integer."""
    if isinstance(label, (int, float)):
        return 1 if label == 1 else 0

    label_str = str(label).strip().lower()

    if label_str in ("yes", "true", "1"):
        return 1
    elif label_str in ("no", "false", "0"):
        return 0
    else:
        return 0  # Default to 0 for unknown


def softmax(logprobs: list) -> list:
    """Calculate softmax from logprobs."""
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    return [p / total for p in probs]


def call_api(
    client: OpenAI, model: str, prompt: str, max_tokens: int, temperature: float
) -> tuple[str, float]:
    """
    Call OpenAI-compatible API with the given prompt.

    Returns:
        tuple: (response_text, yes_probability)
    """
    # Token IDs for Yes/No in Gemma tokenizer
    YES_TOKEN_ID = 10784
    NO_TOKEN_ID = 3771

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
            extra_body={"allowed_token_ids": [YES_TOKEN_ID, NO_TOKEN_ID]},
        )

        response_text = response.choices[0].message.content.strip()

        # Extract logprobs for Yes/No tokens
        yes_logprob = None
        no_logprob = None

        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs

            for item in top_logprobs:
                token_lower = item.token.strip().lower()
                if token_lower == "yes":
                    yes_logprob = item.logprob
                elif token_lower == "no":
                    no_logprob = item.logprob

        # Calculate yes probability using softmax
        if yes_logprob is not None and no_logprob is not None:
            probs = softmax([no_logprob, yes_logprob])
            yes_prob = probs[1]  # Index 1 is yes
        elif yes_logprob is not None:
            yes_prob = math.exp(yes_logprob)
        elif no_logprob is not None:
            yes_prob = 1.0 - math.exp(no_logprob)
        else:
            # Fallback: use text response
            yes_prob = 1.0 if response_text.lower().startswith("yes") else 0.0

        return response_text, yes_prob

    except Exception as e:
        print(f"API Error: {e}")
        return f"ERROR: {e}", 0.5  # Return 0.5 (uncertain) on error


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive metrics for a given threshold."""

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
        mean_confidence_correct = np.mean(
            [
                y_prob[i] if y_pred[i] == 1 else (1 - y_prob[i])
                for i in range(len(y_pred))
                if correct_mask[i]
            ]
        )
    else:
        mean_confidence_correct = 0.0

    if (~correct_mask).sum() > 0:
        mean_confidence_incorrect = np.mean(
            [
                y_prob[i] if y_pred[i] == 1 else (1 - y_prob[i])
                for i in range(len(y_pred))
                if not correct_mask[i]
            ]
        )
    else:
        mean_confidence_incorrect = 0.0

    mean_confidence_overall = np.mean(
        [y_prob[i] if y_pred[i] == 1 else (1 - y_prob[i]) for i in range(len(y_pred))]
    )

    return {
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "nonai_precision": float(precision[0]),
        "nonai_recall": float(recall[0]),
        "nonai_f1": float(f1[0]),
        "ai_precision": float(precision[1]),
        "ai_recall": float(recall[1]),
        "ai_f1": float(f1[1]),
        "mean_confidence_correct": float(mean_confidence_correct),
        "mean_confidence_incorrect": float(mean_confidence_incorrect),
        "mean_confidence_overall": float(mean_confidence_overall),
    }


def predict_and_cache(
    client, model, template, parquet_files, text_column, max_tokens, temperature
):
    """
    Run predictions once on all datasets and cache results.

    Returns:
        DataFrame with columns: text, true_label, ai_probability, dataset_name, response
    """
    print("\n" + "=" * 80)
    print("PREDICTION PHASE (via vLLM API)")
    print("=" * 80)

    all_results = []

    for parquet_path in parquet_files:
        print(f"\n📂 Processing: {parquet_path}")

        # Load parquet file
        df = pd.read_parquet(parquet_path)
        dataset_name = Path(parquet_path).name

        print(f"   Loaded {len(df):,} samples")

        # Validate columns
        if text_column not in df.columns:
            print(f"   ⚠️  ERROR: Column '{text_column}' not found in {dataset_name}")
            print(f"   Available columns: {list(df.columns)}")
            continue

        # Check if labels exist
        has_labels = "label" in df.columns
        if not has_labels:
            print(f"   ⚠️  WARNING: No labels found in {dataset_name}")
            continue

        # Process each row
        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc=f"   Predicting {dataset_name}"
        ):
            text = row[text_column]

            # Fill template
            prompt = template.format(text=text)

            # Call API - returns (text, probability)
            response_text, yes_prob = call_api(
                client, model, prompt, max_tokens, temperature
            )

            # Normalize label
            true_label = normalize_label(row["label"])

            # Store result
            all_results.append(
                {
                    "text": text[:200] + "..." if len(str(text)) > 200 else text,
                    "text_full": text,
                    "true_label": true_label,
                    "ai_probability": yes_prob,
                    "response": response_text,
                    "dataset_name": dataset_name,
                }
            )

    # Create cached DataFrame
    cached_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE")
    print("=" * 80)
    print(f"Total samples processed: {len(cached_df):,}")
    print(f"Datasets: {cached_df['dataset_name'].nunique()}")
    print("=" * 80)

    return cached_df


def evaluate_thresholds(cached_df, thresholds):
    """
    Evaluate performance across multiple thresholds.

    Returns:
        Dictionary with results for each threshold
    """
    print("\n" + "=" * 80)
    print("THRESHOLD EVALUATION PHASE")
    print("=" * 80)

    results = {}

    # Calculate ROC-AUC once (threshold-independent)
    y_true_combined = cached_df["true_label"].values
    y_prob_combined = cached_df["ai_probability"].values

    try:
        roc_auc_combined = roc_auc_score(y_true_combined, y_prob_combined)
        print(f"\n📊 Combined ROC-AUC: {roc_auc_combined:.4f}")
    except Exception as e:
        print(f"\n⚠️  Could not calculate ROC-AUC: {e}")
        roc_auc_combined = None

    # Calculate per-dataset ROC-AUC
    per_dataset_roc_auc = {}
    for dataset_name in cached_df["dataset_name"].unique():
        dataset_df = cached_df[cached_df["dataset_name"] == dataset_name]
        try:
            roc_auc = roc_auc_score(
                dataset_df["true_label"].values, dataset_df["ai_probability"].values
            )
            per_dataset_roc_auc[dataset_name] = float(roc_auc)
            print(f"   {dataset_name}: ROC-AUC = {roc_auc:.4f}")
        except Exception as e:
            per_dataset_roc_auc[dataset_name] = None
            print(f"   {dataset_name}: ROC-AUC = N/A ({e})")

    # Evaluate each threshold
    print("\n" + "-" * 80)
    for threshold in thresholds:
        # print(f"\n📏 Threshold: {threshold}")

        # Apply threshold to get predictions
        cached_df["predicted_label"] = (
            cached_df["ai_probability"] >= threshold
        ).astype(int)

        # Combined metrics
        y_true = cached_df["true_label"].values
        y_pred = cached_df["predicted_label"].values
        y_prob = cached_df["ai_probability"].values

        combined_metrics = calculate_metrics(y_true, y_pred, y_prob)
        # print(f"   Combined Accuracy: {combined_metrics['accuracy']:.4f}")

        # Per-dataset metrics
        per_dataset_metrics = {}
        for dataset_name in cached_df["dataset_name"].unique():
            dataset_df = cached_df[cached_df["dataset_name"] == dataset_name]

            y_true_ds = dataset_df["true_label"].values
            y_pred_ds = dataset_df["predicted_label"].values
            y_prob_ds = dataset_df["ai_probability"].values

            dataset_metrics = calculate_metrics(y_true_ds, y_pred_ds, y_prob_ds)
            per_dataset_metrics[dataset_name] = dataset_metrics

            # print(f"   {dataset_name}: Accuracy = {dataset_metrics['accuracy']:.4f}")

        # Store results
        results[str(threshold)] = {
            "combined": combined_metrics,
            "per_dataset": per_dataset_metrics,
        }

    print("\n" + "=" * 80)
    print("THRESHOLD EVALUATION COMPLETE")
    print("=" * 80)

    return results, roc_auc_combined, per_dataset_roc_auc


def save_results(
    model,
    api_base,
    datasets,
    thresholds,
    results,
    roc_auc_combined,
    per_dataset_roc_auc,
    cached_df,
    output_path,
):
    """Save evaluation results to JSON and CSV files."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine output paths
    if output_path:
        json_path = output_path
        if not json_path.endswith(".json"):
            json_path = json_path + ".json"
    else:
        json_path = f"results_{timestamp}.json"

    csv_path = json_path.replace(".json", "_predictions.csv")

    # Save JSON results
    output_data = {
        "model": model,
        "api_base": api_base,
        "datasets": datasets,
        "thresholds": thresholds,
        "roc_auc_combined": roc_auc_combined,
        "per_dataset_roc_auc": per_dataset_roc_auc,
        "results": results,
        "evaluation_timestamp": timestamp,
        "total_samples": len(cached_df),
        "probability_statistics": {
            "mean": float(cached_df["ai_probability"].mean()),
            "min": float(cached_df["ai_probability"].min()),
            "max": float(cached_df["ai_probability"].max()),
            "std": float(cached_df["ai_probability"].std()),
        },
    }

    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {json_path}")

    # Save predictions CSV
    output_df = cached_df[
        ["dataset_name", "text", "true_label", "ai_probability", "response"]
    ]
    output_df.to_csv(csv_path, index=False)
    print(f"💾 Predictions saved to: {csv_path}")

    return json_path, csv_path


def print_detailed_summary_tables(
    results, thresholds, datasets, per_dataset_roc_auc, roc_auc_combined, cached_df
):
    """Print detailed per-dataset summary tables like evaluate_fasttext.py"""

    # Print per-dataset summary tables
    for dataset_name in datasets:
        dataset_count = len(cached_df[cached_df["dataset_name"] == dataset_name])

        print("\n" + "=" * 80)
        print(f"THRESHOLD SUMMARY: {dataset_name}, total samples: {dataset_count}")
        print("=" * 80)

        # Get ROC-AUC for this dataset
        ds_roc_auc = per_dataset_roc_auc.get(dataset_name)
        if ds_roc_auc is not None:
            print(f"ROC-AUC: {ds_roc_auc:.4f}")
        else:
            print("ROC-AUC: nan")

        print(
            f"\n{'Threshold':<12} {'Accuracy':>10} {'AI Recall':>12} {'AI Prec':>10} {'AI F1':>10} {'NonAI Recall':>14} {'NonAI Prec':>12}"
        )
        print("-" * 95)

        for threshold in thresholds:
            m = results[str(threshold)]["per_dataset"].get(dataset_name, {})
            if m:
                print(
                    f"{threshold:<12.1f} {m['accuracy']:>10.4f} {m['ai_recall']:>12.4f} {m['ai_precision']:>10.4f} {m['ai_f1']:>10.4f} {m['nonai_recall']:>14.4f} {m['nonai_precision']:>12.4f}"
                )

    # Print combined summary table
    print("\n" + "=" * 80)
    print("THRESHOLD EVALUATION SUMMARY (COMBINED)")
    print("=" * 80)
    if roc_auc_combined is not None:
        print(f"Combined ROC-AUC: {roc_auc_combined:.4f}")
    else:
        print("Combined ROC-AUC: nan")

    print(
        f"\n{'Threshold':<12} {'Accuracy':>10} {'AI Recall':>12} {'AI Precision':>14} {'AI F1':>10}"
    )
    print("-" * 60)

    for threshold in thresholds:
        m = results[str(threshold)]["combined"]
        print(
            f"{threshold:<12.1f} {m['accuracy']:>10.4f} {m['ai_recall']:>12.4f} {m['ai_precision']:>14.4f} {m['ai_f1']:>10.4f}"
        )


def main():
    args = parse_args()

    # Setup API configuration
    api_base = args.api_base or os.environ.get("OPENAI_API_BASE")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: API key required. Set OPENAI_API_KEY env var or use --api-key")
        sys.exit(1)

    # Initialize OpenAI client
    client_kwargs = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base

    client = OpenAI(**client_kwargs)

    # Print configuration
    print("=" * 80)
    print("PARQUET API MULTI-THRESHOLD CLASSIFICATION")
    print("=" * 80)
    print(f"Input files: {len(args.input)}")
    for f in args.input:
        print(f"  - {f}")
    print(f"Template file: {args.template}")
    print(f"API base: {api_base or 'default (api.openai.com)'}")
    print(f"Model: {args.model}")
    print(f"Text column: {args.text_column}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Thresholds: {args.thresholds}")
    print("=" * 80)

    # Load template
    template = load_template(args.template)
    print(f"\nTemplate loaded ({len(template)} chars)")

    # Phase 1: Predict and cache all results
    cached_df = predict_and_cache(
        client=client,
        model=args.model,
        template=template,
        parquet_files=args.input,
        text_column=args.text_column,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if len(cached_df) == 0:
        print("ERROR: No samples processed. Check input files and columns.")
        sys.exit(1)

    # Phase 2: Evaluate all thresholds
    results, roc_auc_combined, per_dataset_roc_auc = evaluate_thresholds(
        cached_df, args.thresholds
    )

    # Get dataset names
    dataset_names = list(cached_df["dataset_name"].unique())

    # Print detailed summary tables (per-dataset + combined)
    print_detailed_summary_tables(
        results,
        args.thresholds,
        dataset_names,
        per_dataset_roc_auc,
        roc_auc_combined,
        cached_df,
    )

    # Save results
    json_path, csv_path = save_results(
        model=args.model,
        api_base=api_base,
        datasets=dataset_names,
        thresholds=args.thresholds,
        results=results,
        roc_auc_combined=roc_auc_combined,
        per_dataset_roc_auc=per_dataset_roc_auc,
        cached_df=cached_df,
        output_path=args.output,
    )

    print("\n✅ Done!")
    print(f"   JSON results: {json_path}")
    print(f"   Predictions CSV: {csv_path}")

    return results, cached_df


if __name__ == "__main__":
    main()
