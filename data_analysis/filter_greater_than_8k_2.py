#!/usr/bin/env python3

"""
Filter parquet files to keep only rows with token_length <= 8000.
Saves filtered files with _8k suffix.
"""

import os
import glob
import argparse
import pandas as pd
from tqdm.auto import tqdm

# Default configuration
DEFAULT_INPUT_DIR = "./data_analysis/data/token_data_and_label_2"
DEFAULT_OUTPUT_DIR = "./data_analysis/data/filtered_less_8k_3"
MAX_TOKEN_LENGTH = 8000

def filter_parquet_file(input_path: str, output_path: str, max_tokens: int = MAX_TOKEN_LENGTH):
    """
    Load parquet, filter rows with token_length <= max_tokens, save to output path.
    """
    filename = os.path.basename(input_path)
    print(f"\n📄 Processing: {filename}")
    
    # Load parquet
    df = pd.read_parquet(input_path)
    
    if "token_length" not in df.columns:
        print(f"  ⚠️  Skipping - no 'token_length' column found")
        return
    
    # Original stats
    original_len = len(df)
    original_mean = df["token_length"].mean()
    original_max = df["token_length"].max()
    rows_over_limit = (df["token_length"] > max_tokens).sum()
    
    print(f"  ℹ️  Original rows: {original_len:,}")
    print(f"  📊 Original token stats: mean={original_mean:.0f}, max={original_max}")
    print(f"  🔍 Rows with token_length > {max_tokens}: {rows_over_limit:,}")
    
    # Filter: keep only rows with token_length <= max_tokens
    df_filtered = df[df["token_length"] <= max_tokens].copy()
    
    # Filtered stats
    filtered_len = len(df_filtered)
    filtered_mean = df_filtered["token_length"].mean()
    filtered_max = df_filtered["token_length"].max()
    removed_count = original_len - filtered_len
    removed_pct = (removed_count / original_len * 100) if original_len > 0 else 0
    
    print(f"  ✂️  Filtered rows: {filtered_len:,} (removed {removed_count:,} rows, {removed_pct:.1f}%)")
    print(f"  📊 Filtered token stats: mean={filtered_mean:.0f}, max={filtered_max}")
    
    # Save to output directory
    df_filtered.to_parquet(output_path, index=False)
    print(f"  ✅ Saved → {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Filter parquet files to keep only rows with token_length <= 8000"
    )
    parser.add_argument(
        "--input_dir", 
        default=DEFAULT_INPUT_DIR,
        help="Input directory containing parquet files with token_length column"
    )
    parser.add_argument(
        "--output_dir", 
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for filtered parquet files"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=MAX_TOKEN_LENGTH,
        help="Maximum token length to keep (default: 8000)"
    )
    parser.add_argument(
        "--glob", 
        default="*.parquet",
        help="Glob pattern for input files"
    )
    
    args = parser.parse_args()
    
    # Find input files
    input_files = sorted(glob.glob(os.path.join(args.input_dir, args.glob)))
    
    if not input_files:
        print(f"❌ No parquet files found in {args.input_dir}")
        return
    
    print(f"🔍 Found {len(input_files)} file(s)")
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"✂️  Max token length: {args.max_tokens}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each file
    print("\n" + "="*80)
    for input_path in input_files:
        # Get filename and add _8k suffix
        filename = os.path.basename(input_path)
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{name_without_ext}_8k.parquet"
        output_path = os.path.join(args.output_dir, output_filename)
        
        try:
            filter_parquet_file(input_path, output_path, args.max_tokens)
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print("\n" + "="*80)
    print("✅ All files processed!")
    print(f"📂 Output location: {args.output_dir}")

if __name__ == "__main__":
    main()