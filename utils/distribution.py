# From host machine
import pandas as pd
labels_df = pd.read_csv('/home/user/data/all_data/labels.csv')
print("\n" + "="*50)
print("ACTUAL CLASS DISTRIBUTION")
print("="*50)
print("\nCounts:")
print(labels_df['label'].value_counts().sort_index())
print("\nPercentages:")
print(labels_df['label'].value_counts(normalize=True).sort_index() * 100)
total = labels_df['label'].value_counts()
print(f"\nImbalance Ratio: {total.max()}/{total.min()} = {total.max()/total.min():.2f}:1")
