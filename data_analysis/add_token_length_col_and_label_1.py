#!/usr/bin/env python3

"""
Simple script to add token_length and label columns to parquet files.
- Adds token_length by tokenizing the text
- Adds label: 1 if filename contains 'positive', 0 if contains 'negative'
- Drops unnecessary columns: dump, url, date (if they exist)
"""

import os
import glob
import argparse
import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# Default configuration
DEFAULT_DATA_DIR = "./data_analysis/data/raw_data_1"
DEFAULT_OUTPUT_DIR = "./data_analysis/data/token_data_and_label_2"
TOKENIZER_NAME = "google/gemma-3-1b-it"
BATCH_SIZE = 256

# Columns to drop if they exist
COLUMNS_TO_DROP = ['dump', 'url', 'date']

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def determine_label_from_filename(filename: str) -> int:
    """
    Determine label from filename:
    - 1 if 'positive' in filename
    - 0 if 'negative' in filename
    - Raises error if neither found
    """
    filename_lower = filename.lower()
    
    if 'positive' in filename_lower:
        return 1
    elif 'negative' in filename_lower:
        return 0
    else:
        raise ValueError(
            f"Filename '{filename}' must contain 'positive' or 'negative' to determine label"
        )

def add_token_lengths(input_path: str, output_path: str, tokenizer, batch_size: int = BATCH_SIZE):
    """
    Load parquet, add token_length and label columns, save to output path.
    """
    filename = os.path.basename(input_path)
    print(f"\n📄 Processing: {filename}")
    
    # Determine label from filename
    try:
        label = determine_label_from_filename(filename)
        label_name = "AI (positive)" if label == 1 else "Non-AI (negative)"
        print(f"  🏷️  Label: {label} ({label_name})")
    except ValueError as e:
        print(f"  ❌ {e}")
        return
    
    # Load parquet
    df = pd.read_parquet(input_path)
    
    if "text" not in df.columns:
        print(f"  ⚠️  Skipping - no 'text' column found")
        return
    
    # Drop unnecessary columns if they exist
    columns_dropped = []
    for col in COLUMNS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])
            columns_dropped.append(col)
    
    if columns_dropped:
        print(f"  🗑️  Dropped columns: {', '.join(columns_dropped)}")
    
    # Clean: drop rows with NaN or empty text
    original_len = len(df)
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.len() > 0]
    
    if len(df) < original_len:
        print(f"  ℹ️  Cleaned: {original_len:,} → {len(df):,} rows")
    else:
        print(f"  ℹ️  Rows: {len(df):,}")
    
    # Add label column (same value for all rows in this file)
    df["label"] = label
    
    # Tokenize in batches
    texts = df["text"].tolist()
    token_lengths = []
    
    for i in tqdm(range(0, len(texts), batch_size), 
                  desc=f"  Tokenizing", 
                  unit="batch",
                  leave=False):
        batch = texts[i:i + batch_size]
        outputs = tokenizer(
            batch, 
            add_special_tokens=False, 
            truncation=False, 
            return_length=True
        )
        token_lengths.extend([int(x) for x in outputs["length"]])
    
    # Add token_length column
    df["token_length"] = token_lengths
    
    # Stats
    mean_tokens = df["token_length"].mean()
    median_tokens = df["token_length"].median()
    max_tokens = df["token_length"].max()
    
    print(f"  📊 Token stats: mean={mean_tokens:.0f}, median={median_tokens:.0f}, max={max_tokens}")
    print(f"  🏷️  Label column: all rows = {label}")
    print(f"  📋 Final columns: {', '.join(df.columns)}")
    
    # Save to output directory
    df.to_parquet(output_path, index=False)
    print(f"  ✅ Saved → {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Add token_length and label columns to parquet files"
    )
    parser.add_argument(
        "--data_dir", 
        default=DEFAULT_DATA_DIR,
        help="Input directory containing parquet files"
    )
    parser.add_argument(
        "--output_dir", 
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for parquet files with token_length and label"
    )
    parser.add_argument(
        "--glob", 
        default="*.parquet",
        help="Glob pattern for input files"
    )
    parser.add_argument(
        "--tokenizer", 
        default=TOKENIZER_NAME,
        help="Tokenizer model name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for tokenization"
    )
    
    args = parser.parse_args()
    
    # Find input files
    input_files = sorted(glob.glob(os.path.join(args.data_dir, args.glob)))
    
    if not input_files:
        print(f"❌ No parquet files found in {args.data_dir}")
        return
    
    print(f"🔍 Found {len(input_files)} file(s)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
    # Load tokenizer
    print(f"🤖 Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    
    # Process each file
    print("\n" + "="*80)
    for input_path in input_files:
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, filename)
        
        try:
            add_token_lengths(input_path, output_path, tokenizer, args.batch_size)
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print("\n" + "="*80)
    print("✅ All files processed!")
    print(f"📂 Output location: {args.output_dir}")

if __name__ == "__main__":
    main()