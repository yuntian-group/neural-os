import pandas as pd
import numpy as np
import re

def extract_record_num(path):
    """Extract record number from any path in the row"""
    match = re.search(r'record_(\d+)', path)
    return int(match.group(1)) if match else None

# Read the CSV file
input_file = 'desktop_sequences_filtered_with_desktop_1.5k.challenging.csv'  # replace with your filename
df = pd.read_csv(input_file)

# Initialize test set indices
test_indices = set()

# Sample 200 rows for each cluster type
cluster_types = ['desktop_firefox', 'desktop_terminal', 'desktop_root', 'desktop_trash']
for cluster_type in cluster_types:
    cluster_df = df[df['matched_cluster'].str.contains(cluster_type, na=False)]
    if len(cluster_df) > 200:
        sampled_indices = cluster_df.sample(n=200, random_state=42).index
        test_indices.update(sampled_indices)

# Extract record numbers from sampled rows
test_records = set()
for idx in test_indices:
    row = df.iloc[idx]
    # Extract from Image_seq_cond_path
    paths = eval(row['Image_seq_cond_path'])
    for path in paths:
        record_num = extract_record_num(path)
        if record_num:
            test_records.add(record_num)
    # Extract from Target_image
    record_num = extract_record_num(row['Target_image'])
    if record_num:
        test_records.add(record_num)

# Add all rows with matching record numbers to test set
for idx, row in df.iterrows():
    paths = eval(row['Image_seq_cond_path'])
    for path in paths:
        record_num = extract_record_num(path)
        if record_num in test_records:
            test_indices.add(idx)
            break
    if idx not in test_indices:
        record_num = extract_record_num(row['Target_image'])
        if record_num in test_records:
            test_indices.add(idx)

# Create test and train dataframes
test_df = df.loc[list(test_indices)]
train_df = df.drop(list(test_indices))

# Shuffle both dataframes
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to files
base_name = input_file.rsplit('.', 1)[0]
train_df.to_csv(f'{base_name}.train.csv', index=False)
test_df.to_csv(f'{base_name}.test.csv', index=False)

# Print statistics
print(f"Total rows: {len(df)}")
print(f"Train rows: {len(train_df)}")
print(f"Test rows: {len(test_df)}")
print("\nTest set cluster distribution:")
for cluster_type in cluster_types:
    count = len(test_df[test_df['matched_cluster'].str.contains(cluster_type, na=False)])
    print(f"{cluster_type}: {count}")
print(f"\nUnique record numbers in test set: {len(test_records)}")
