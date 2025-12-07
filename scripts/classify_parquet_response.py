#!/usr/bin/env python3
"""
Parquet API Classification Script

Iterates through a parquet file, fills a template with each row's text,
sends it to an OpenAI-compatible API, and evaluates yes/no responses
against the label column.
"""

import argparse
import os
import re
import sys
from datetime import datetime

import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process parquet file through OpenAI-compatible API and evaluate responses"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input parquet file"
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
        default="",
        help="API key (overrides OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stage-1-classifier",
        help="Model name to use (default: stage-1-classifier)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: results_<timestamp>.csv)",
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
        help="Maximum tokens for API response (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for API response (default: 0.0)",
    )
    return parser.parse_args()


def load_config(self):
    """Load configuration from YAML"""
    with open("config/instruction_training_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def load_template(template_path: str) -> str:
    """Load template from file."""
    config = load_config("config/instruction_training_config.yaml")
    template = config["instruction"]["prompt_template"]

    if "{text}" not in template:
        print(f"WARNING: Template does not contain {{text}} placeholder")

    return template


def normalize_label(label) -> str:
    """
    Normalize label to 'yes' or 'no' string.

    Handles:
    - 0/1 integers -> 'no'/'yes'
    - 'yes'/'no' strings (case-insensitive)
    - 'true'/'false' strings -> 'yes'/'no'
    """
    if isinstance(label, (int, float)):
        return "yes" if label == 1 else "no"

    label_str = str(label).strip().lower()

    if label_str in ("yes", "true", "1"):
        return "yes"
    elif label_str in ("no", "false", "0"):
        return "no"
    else:
        return label_str


def extract_yes_no(response: str) -> str:
    """
    Extract yes/no from API response.

    Returns 'yes', 'no', or 'unknown' if neither found.
    """
    response_lower = response.strip().lower()

    # Check for exact match first
    if response_lower in ("yes", "no"):
        return response_lower

    # Check if response starts with yes/no
    if response_lower.startswith("yes"):
        return "yes"
    if response_lower.startswith("no"):
        return "no"

    # Search for yes/no anywhere in response
    yes_match = re.search(r"\byes\b", response_lower)
    no_match = re.search(r"\bno\b", response_lower)

    if yes_match and not no_match:
        return "yes"
    elif no_match and not yes_match:
        return "no"
    elif yes_match and no_match:
        # Both found - use the one that appears first
        return "yes" if yes_match.start() < no_match.start() else "no"

    return "unknown"


def call_api(
    client: OpenAI, model: str, prompt: str, max_tokens: int, temperature: float
) -> str:
    """
    Call OpenAI-compatible API with the given prompt.

    Returns the response text.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
            extra_body={"allowed_token_ids": [10784, 3771]},
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return f"ERROR: {e}"


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
    print("PARQUET API CLASSIFICATION")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Template file: {args.template}")
    print(f"API base: {api_base or 'default (api.openai.com)'}")
    print(f"Model: {args.model}")
    print(f"Text column: {args.text_column}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print("=" * 80)

    # Load template
    template = load_template(args.template)
    print(f"\nTemplate loaded ({len(template)} chars)")

    # Load parquet file
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows from parquet file")

    # Validate columns
    if args.text_column not in df.columns:
        print(f"ERROR: Column '{args.text_column}' not found in parquet file")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    has_labels = "label" in df.columns
    if has_labels:
        print("Label column found - will calculate accuracy")
    else:
        print("WARNING: No 'label' column found - will only show predictions")

    # Process each row
    results = []
    correct_count = 0
    incorrect_count = 0
    unknown_count = 0

    print(f"\nProcessing {len(df):,} rows...")
    print("-" * 80)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        text = row[args.text_column]

        # Fill template
        prompt = template.format(text=text)

        # Call API
        response = call_api(
            client, args.model, prompt, args.max_tokens, args.temperature
        )

        # Extract yes/no from response
        extracted = extract_yes_no(response)

        # Compare with label if available
        is_correct = None
        expected = None

        if has_labels:
            expected = normalize_label(row["label"])
            if extracted == "unknown":
                is_correct = None
                unknown_count += 1
            elif extracted == expected:
                is_correct = True
                correct_count += 1
            else:
                is_correct = False
                incorrect_count += 1

        # Store result
        result = {
            "index": idx,
            "text": text[:200] + "..." if len(str(text)) > 200 else text,
            "text_full": text,
            "label": expected,
            "response": response,
            "extracted": extracted,
            "is_correct": is_correct,
        }
        results.append(result)

        # Print per-row result
        status = ""
        if is_correct is True:
            status = "CORRECT"
        elif is_correct is False:
            status = "INCORRECT"
        elif extracted == "unknown":
            status = "UNKNOWN"
        else:
            status = "N/A"

        tqdm.write(
            f"[{idx}] Response: {response[:50]}... | Extracted: {extracted} | Expected: {expected} | {status}"
        )

    print("-" * 80)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total rows processed: {len(results_df):,}")

    if has_labels:
        total_evaluated = correct_count + incorrect_count
        if total_evaluated > 0:
            accuracy = correct_count / total_evaluated * 100
            print(f"\nAccuracy: {accuracy:.2f}% ({correct_count}/{total_evaluated})")
        print(f"  Correct: {correct_count:,}")
        print(f"  Incorrect: {incorrect_count:,}")
        print(f"  Unknown/Unparseable: {unknown_count:,}")

    # Count response distribution
    print(f"\nResponse distribution:")
    for val in ["yes", "no", "unknown"]:
        count = (results_df["extracted"] == val).sum()
        pct = count / len(results_df) * 100
        print(f"  {val}: {count:,} ({pct:.1f}%)")

    print("=" * 80)

    # Save results to CSV
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results_{timestamp}.csv"

    # Save with selected columns
    output_df = results_df[
        ["index", "text", "label", "response", "extracted", "is_correct"]
    ]
    output_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Also save full text version
    full_output_path = output_path.replace(".csv", "_full.csv")
    results_df.to_csv(full_output_path, index=False)
    print(f"Full results saved to: {full_output_path}")

    print("\nDone!")

    return results_df


if __name__ == "__main__":
    main()
