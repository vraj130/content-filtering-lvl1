#!/usr/bin/env python3
"""
Split parquet files into train/val/test sets with stratified sampling.
- Takes 20k samples from each file (positive and negative)
- Creates 80/10/10 split (32k/4k/4k total)
- Saves remaining samples separately
"""

import os
import pandas as pd
import numpy as np

# Configuration
INPUT_DIR = "./data_analysis/data/token_data_and_label_2"
OUTPUT_TRAIN_DIR = "./data_analysis/data/final_data_4/train_all_data"
OUTPUT_REMAINING_DIR = "./data_analysis/data/final_data_4/in_distribution_test_data"

POSITIVE_FILE = "alignment_dataset_ai_positive.parquet"
NEGATIVE_FILE = "alignment_dataset_ai_negative.parquet"

SAMPLES_PER_CLASS = 20000
RANDOM_SEED = 42

# Split ratios
TRAIN_SIZE = 16000  # per class
VAL_SIZE = 2000     # per class
TEST_SIZE = 2000    # per class

def main():
    print("="*80)
    print("Creating Train/Val/Test Split")
    print("="*80)
    
    # Create output directories
    os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUTPUT_REMAINING_DIR, exist_ok=True)
    print(f"\n📁 Output directories created:")
    print(f"   Train/Val/Test: {OUTPUT_TRAIN_DIR}")
    print(f"   Remaining data: {OUTPUT_REMAINING_DIR}")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # ========== Process Positive File ==========
    print(f"\n{'='*80}")
    print(f"Processing POSITIVE file (label=1)")
    print(f"{'='*80}")
    
    positive_path = os.path.join(INPUT_DIR, POSITIVE_FILE)
    print(f"Loading: {positive_path}")
    df_positive = pd.read_parquet(positive_path)
    print(f"Total samples: {len(df_positive):,}")
    
    # Shuffle
    df_positive = df_positive.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"✓ Shuffled with seed={RANDOM_SEED}")
    
    # Split: first 20k for train/val/test, rest for remaining
    df_positive_selected = df_positive.iloc[:SAMPLES_PER_CLASS].copy()
    df_positive_remaining = df_positive.iloc[SAMPLES_PER_CLASS:].copy()
    
    print(f"✓ Selected: {len(df_positive_selected):,} samples for train/val/test")
    print(f"✓ Remaining: {len(df_positive_remaining):,} samples")
    
    # Split selected data into train/val/test
    pos_train = df_positive_selected.iloc[:TRAIN_SIZE].copy()
    pos_val = df_positive_selected.iloc[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE].copy()
    pos_test = df_positive_selected.iloc[TRAIN_SIZE+VAL_SIZE:TRAIN_SIZE+VAL_SIZE+TEST_SIZE].copy()
    
    print(f"✓ Split: Train={len(pos_train):,}, Val={len(pos_val):,}, Test={len(pos_test):,}")
    
    # ========== Process Negative File ==========
    print(f"\n{'='*80}")
    print(f"Processing NEGATIVE file (label=0)")
    print(f"{'='*80}")
    
    negative_path = os.path.join(INPUT_DIR, NEGATIVE_FILE)
    print(f"Loading: {negative_path}")
    df_negative = pd.read_parquet(negative_path)
    print(f"Total samples: {len(df_negative):,}")
    
    # Shuffle
    df_negative = df_negative.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"✓ Shuffled with seed={RANDOM_SEED}")
    
    # Split: first 20k for train/val/test, rest for remaining
    df_negative_selected = df_negative.iloc[:SAMPLES_PER_CLASS].copy()
    df_negative_remaining = df_negative.iloc[SAMPLES_PER_CLASS:].copy()
    
    print(f"✓ Selected: {len(df_negative_selected):,} samples for train/val/test")
    print(f"✓ Remaining: {len(df_negative_remaining):,} samples")
    
    # Split selected data into train/val/test
    neg_train = df_negative_selected.iloc[:TRAIN_SIZE].copy()
    neg_val = df_negative_selected.iloc[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE].copy()
    neg_test = df_negative_selected.iloc[TRAIN_SIZE+VAL_SIZE:TRAIN_SIZE+VAL_SIZE+TEST_SIZE].copy()
    
    print(f"✓ Split: Train={len(neg_train):,}, Val={len(neg_val):,}, Test={len(neg_test):,}")
    
    # ========== Combine and Save Train/Val/Test ==========
    print(f"\n{'='*80}")
    print(f"Combining and Saving Train/Val/Test Sets")
    print(f"{'='*80}")
    
    # Combine positive and negative
    train_df = pd.concat([pos_train, neg_train], ignore_index=True)
    val_df = pd.concat([pos_val, neg_val], ignore_index=True)
    test_df = pd.concat([pos_test, neg_test], ignore_index=True)
    
    # Shuffle combined datasets
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"\n📊 Final Dataset Sizes:")
    print(f"   Train: {len(train_df):,} samples ({(train_df['label']==1).sum():,} AI, {(train_df['label']==0).sum():,} non-AI)")
    print(f"   Val:   {len(val_df):,} samples ({(val_df['label']==1).sum():,} AI, {(val_df['label']==0).sum():,} non-AI)")
    print(f"   Test:  {len(test_df):,} samples ({(test_df['label']==1).sum():,} AI, {(test_df['label']==0).sum():,} non-AI)")
    
    # Save train/val/test
    train_path = os.path.join(OUTPUT_TRAIN_DIR, "train.parquet")
    val_path = os.path.join(OUTPUT_TRAIN_DIR, "val.parquet")
    test_path = os.path.join(OUTPUT_TRAIN_DIR, "test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"\n✅ Saved:")
    print(f"   {train_path}")
    print(f"   {val_path}")
    print(f"   {test_path}")
    
    # ========== Save Remaining Data ==========
    print(f"\n{'='*80}")
    print(f"Saving Remaining Data (Out-of-Distribution Test Set)")
    print(f"{'='*80}")
    
    remaining_pos_path = os.path.join(OUTPUT_REMAINING_DIR, POSITIVE_FILE)
    remaining_neg_path = os.path.join(OUTPUT_REMAINING_DIR, NEGATIVE_FILE)
    
    df_positive_remaining.to_parquet(remaining_pos_path, index=False)
    df_negative_remaining.to_parquet(remaining_neg_path, index=False)
    
    print(f"\n📊 Remaining Dataset Sizes:")
    print(f"   Positive (AI): {len(df_positive_remaining):,} samples")
    print(f"   Negative (non-AI): {len(df_negative_remaining):,} samples")
    print(f"   Total: {len(df_positive_remaining) + len(df_negative_remaining):,} samples")
    
    print(f"\n✅ Saved:")
    print(f"   {remaining_pos_path}")
    print(f"   {remaining_neg_path}")
    
    # ========== Summary ==========
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE - Summary")
    print(f"{'='*80}")
    print(f"\nTrain/Val/Test (balanced, shuffled):")
    print(f"   Location: {OUTPUT_TRAIN_DIR}")
    print(f"   Total: {len(train_df) + len(val_df) + len(test_df):,} samples")
    
    print(f"\nRemaining data (for future testing):")
    print(f"   Location: {OUTPUT_REMAINING_DIR}")
    print(f"   Total: {len(df_positive_remaining) + len(df_negative_remaining):,} samples")
    
    print(f"\nRandom seed used: {RANDOM_SEED}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()