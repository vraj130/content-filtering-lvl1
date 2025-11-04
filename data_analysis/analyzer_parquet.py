import pandas as pd
import sys
import os

def quick_parquet_analysis(parquet_path):
    print(f"\n📄 File: {os.path.basename(parquet_path)}")
    print("="*60)
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"❌ Could not read parquet: {e}")
        return

    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns\n")

    print("🔍 Columns:")
    print(df.dtypes)
    print("\nSample rows:")
    print(df.head(3))

    print("\n📊 Null values per column:")
    print(df.isnull().sum())

    print("\n📊 Basic stats:\n")
    with pd.option_context('display.max_columns', None):
        print(df.describe(include='all').transpose())

    print("\n📊 Example value counts for categorical columns:")
    for col in df.select_dtypes(include=["object", "category"]).columns:
        print(f" - '{col}':")
        print(df[col].value_counts().head(5), "\n")

    if "token_length" in df.columns:
        print("\nℹ️ token_length stats:")
        print(df["token_length"].describe())
    if "label" in df.columns:
        print("\nℹ️ label counts:")
        print(df["label"].value_counts())
    print("="*60)

if __name__ == "__main__":
    quick_parquet_analysis("./data_analysis/data/final_data_4/ood_distrbution_test_data/fineweb_ai_negative_8k.parquet")
