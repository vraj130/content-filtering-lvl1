import pandas as pd
import re
from collections import Counter
from tqdm import tqdm
import json

# File paths
AI_FILE = "./data_analysis/data/final_data_4/ood_distrbution_test_data/fineweb_ai_positive_8k.parquet"
NON_AI_FILE = "./data_analysis/data/final_data_4/ood_distrbution_test_data/fineweb_ai_negative_8k.parquet"

# Output files
OUTPUT_FILTERED = "ai_keyword_analysis_filtered.json"
OUTPUT_ALL_RATIOS = "ai_keyword_analysis_all_ratios.json"

# Parameters
RATIO_THRESHOLD =   # Words must be 10x more common in AI docs
MIN_AI_FREQ = 0.01    # Must appear in at least 1% of AI docs
MIN_WORD_LENGTH = 3   # Ignore very short words

print("=" * 70)
print("AI vs Non-AI Keyword Frequency Analysis")
print("Using Document-Level Frequency (Presence/Absence)")
print("=" * 70)

# Load datasets
print("\n[1/6] Loading AI documents...")
df_ai = pd.read_parquet(AI_FILE)
print(f"   ✓ Loaded {len(df_ai):,} AI documents")

print("\n[2/6] Loading Non-AI documents...")
df_nonai = pd.read_parquet(NON_AI_FILE)
print(f"   ✓ Loaded {len(df_nonai):,} Non-AI documents")

n_ai_docs = len(df_ai)
n_nonai_docs = len(df_nonai)

# Tokenization function
def tokenize(text):
    """Extract lowercase alphabetic words of length >= MIN_WORD_LENGTH"""
    if pd.isna(text):
        return []
    words = re.findall(r'\b[a-z]+\b', str(text).lower())
    return [w for w in words if len(w) >= MIN_WORD_LENGTH]

# Count DOCUMENT frequency in AI documents (each word counted once per document)
print(f"\n[3/6] Counting document frequencies in AI documents...")
ai_word_doc_counts = Counter()
for text in tqdm(df_ai['text'], desc="   Processing AI docs", unit="doc"):
    words = tokenize(text)
    unique_words = set(words)  # Each word counted once per document
    ai_word_doc_counts.update(unique_words)

print(f"   ✓ Found {len(ai_word_doc_counts):,} unique words in AI documents")

# Count DOCUMENT frequency in Non-AI documents (each word counted once per document)
print(f"\n[4/6] Counting document frequencies in Non-AI documents...")
nonai_word_doc_counts = Counter()
for text in tqdm(df_nonai['text'], desc="   Processing Non-AI docs", unit="doc"):
    words = tokenize(text)
    unique_words = set(words)  # Each word counted once per document
    nonai_word_doc_counts.update(unique_words)

print(f"   ✓ Found {len(nonai_word_doc_counts):,} unique words in Non-AI documents")

# Calculate averaged document frequencies (what % of documents contain each word)
print("\n[5/6] Calculating frequency ratios...")
avg_freq_ai = {word: count / n_ai_docs for word, count in ai_word_doc_counts.items()}
avg_freq_nonai = {word: count / n_nonai_docs for word, count in nonai_word_doc_counts.items()}

# Get common vocabulary
common_vocab = set(avg_freq_ai.keys()) & set(avg_freq_nonai.keys())
print(f"   ✓ Found {len(common_vocab):,} words appearing in both classes")

# Calculate ratios for ALL common words
all_ratios = []
for word in tqdm(common_vocab, desc="   Computing ratios", unit="word"):
    ratio = avg_freq_ai[word] / avg_freq_nonai[word]
    all_ratios.append({
        'word': word,
        'ratio': ratio,
        'doc_freq_ai': avg_freq_ai[word],
        'doc_freq_nonai': avg_freq_nonai[word],
        'num_docs_ai': ai_word_doc_counts[word],
        'num_docs_nonai': nonai_word_doc_counts[word]
    })

# Sort all ratios by ratio value
all_ratios.sort(key=lambda x: x['ratio'], reverse=True)

# Apply filters for the filtered list
print("\n[6/6] Applying filters and ranking keywords...")
filtered_keywords = []
for item in all_ratios:
    ratio = item['ratio']
    ai_freq = item['doc_freq_ai']
    
    # Apply threshold filters
    if ratio >= RATIO_THRESHOLD and ai_freq >= MIN_AI_FREQ:
        filtered_keywords.append(item)

print(f"   ✓ Found {len(filtered_keywords):,} AI-discriminative keywords (after filtering)")
print(f"   ✓ Total words analyzed: {len(all_ratios):,}")

# Display results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nDataset sizes:")
print(f"  - AI documents: {n_ai_docs:,}")
print(f"  - Non-AI documents: {n_nonai_docs:,}")

print(f"\nFilters applied:")
print(f"  - Minimum ratio threshold: {RATIO_THRESHOLD}x")
print(f"  - Minimum AI doc frequency: {MIN_AI_FREQ} ({MIN_AI_FREQ * n_ai_docs:.0f} docs)")
print(f"  - Minimum word length: {MIN_WORD_LENGTH} characters")

print(f"\n{'='*70}")
print(f"Top 50 AI-Discriminative Keywords (After Filtering)")
print(f"{'='*70}")
print(f"{'Rank':<6} {'Word':<20} {'Ratio':<10} {'AI %':<10} {'Non-AI %':<12} {'AI Docs':<10}")
print("-" * 70)

for i, kw in enumerate(filtered_keywords[:50], 1):
    ai_pct = kw['doc_freq_ai'] * 100
    nonai_pct = kw['doc_freq_nonai'] * 100
    print(f"{i:<6} {kw['word']:<20} {kw['ratio']:>8.1f}x  "
          f"{ai_pct:>7.1f}%   {nonai_pct:>9.2f}%   {kw['num_docs_ai']:>8,}")

# Save filtered results to JSON
output_filtered_data = {
    'metadata': {
        'n_ai_docs': n_ai_docs,
        'n_nonai_docs': n_nonai_docs,
        'ratio_threshold': RATIO_THRESHOLD,
        'min_ai_freq': MIN_AI_FREQ,
        'min_word_length': MIN_WORD_LENGTH,
        'total_ai_keywords_found': len(filtered_keywords),
        'total_words_analyzed': len(all_ratios),
        'method': 'document_level_frequency'
    },
    'keywords': filtered_keywords
}

with open(OUTPUT_FILTERED, 'w') as f:
    json.dump(output_filtered_data, f, indent=2)

print(f"\n✓ Filtered results saved to: {OUTPUT_FILTERED}")

# Save ALL ratios to JSON (unfiltered)
output_all_data = {
    'metadata': {
        'n_ai_docs': n_ai_docs,
        'n_nonai_docs': n_nonai_docs,
        'min_word_length': MIN_WORD_LENGTH,
        'total_words_analyzed': len(all_ratios),
        'method': 'document_level_frequency',
        'note': 'This file contains ALL words with their ratios (no filtering applied)'
    },
    'all_word_ratios': all_ratios
}

with open(OUTPUT_ALL_RATIOS, 'w') as f:
    json.dump(output_all_data, f, indent=2)

print(f"✓ All word ratios saved to: {OUTPUT_ALL_RATIOS}")

# Additional statistics
print(f"\n{'='*70}")
print("Additional Statistics")
print(f"{'='*70}")

if filtered_keywords:
    ratios_list = [kw['ratio'] for kw in filtered_keywords]
    print(f"\nFiltered Keywords:")
    print(f"  - Highest ratio: {max(ratios_list):.1f}x ({filtered_keywords[0]['word']})")
    print(f"  - Median ratio: {sorted(ratios_list)[len(ratios_list)//2]:.1f}x")
    print(f"  - Average ratio: {sum(ratios_list)/len(ratios_list):.1f}x")
    
    # Show example interpretations
    print(f"\nExample interpretations:")
    for kw in filtered_keywords[:3]:
        ai_pct = kw['doc_freq_ai'] * 100
        nonai_pct = kw['doc_freq_nonai'] * 100
        print(f"  '{kw['word']}':")
        print(f"    - Appears in {ai_pct:.1f}% of AI docs ({kw['num_docs_ai']:,} docs)")
        print(f"    - Appears in {nonai_pct:.2f}% of Non-AI docs ({kw['num_docs_nonai']:,} docs)")
        print(f"    - {kw['ratio']:.1f}x more common in AI documents\n")
    
    # Show distribution of keywords by ratio ranges
    print(f"Keyword distribution by ratio (filtered):")
    ranges = [(10, 20), (20, 50), (50, 100), (100, float('inf'))]
    for low, high in ranges:
        count = sum(1 for r in ratios_list if low <= r < high)
        upper = "+" if high == float('inf') else str(int(high))
        print(f"  - {low}x - {upper}x: {count:,} keywords")

# Statistics for ALL ratios
all_ratios_values = [item['ratio'] for item in all_ratios]
print(f"\nAll Words Statistics:")
print(f"  - Total unique words in both classes: {len(all_ratios):,}")
print(f"  - Highest ratio: {max(all_ratios_values):.1f}x ({all_ratios[0]['word']})")
print(f"  - Lowest ratio: {min(all_ratios_values):.4f}x ({all_ratios[-1]['word']})")
print(f"  - Median ratio: {sorted(all_ratios_values)[len(all_ratios_values)//2]:.2f}x")

print("\n" + "=" * 70)
print("Analysis complete!")
print("=" * 70)