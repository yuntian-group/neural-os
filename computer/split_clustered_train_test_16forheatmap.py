import pandas as pd
import numpy as np
from collections import defaultdict

def split_by_clusters(input_csv, train_csv, test_csv, samples_per_cluster=5):
    """
    Split dataset into train and test sets, taking N samples per cluster for test
    while ensuring no record_num overlap between train and test sets
    """
    # Load the dataset
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")
    
    # Group by cluster_id
    clusters = defaultdict(list)
    for _, row in df.iterrows():
        clusters[row['cluster_id']].append(row.to_dict())
    
    # Select test samples
    test_rows = []
    used_records = set()  # Keep track of used record_nums
    
    print(f"Selecting {samples_per_cluster} samples per cluster for test set...")
    for cluster_id, rows in clusters.items():
        if cluster_id > 15:
            continue
        # Shuffle rows for this cluster
        cluster_rows = list(np.random.permutation(rows))
        selected = 0
        
        # Try to select samples_per_cluster rows that don't share record_nums with already selected rows
        for row in cluster_rows:
            if selected >= samples_per_cluster:
                break
                
            if row['record_num'] not in used_records:
                test_rows.append(row)
                used_records.add(row['record_num'])
                selected += 1
        
        print(f"Cluster {cluster_id}: selected {selected}/{samples_per_cluster} samples")
    
    # Create test DataFrame
    test_df = pd.DataFrame(test_rows)
    
    # Create train DataFrame (excluding all record_nums that appear in test)
    train_df = df[~df['record_num'].isin(used_records)]
    
    # Save splits
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    # Print statistics
    print("\nSplit statistics:")
    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    
    print("\nTrain cluster distribution:")
    train_counts = train_df['cluster_id'].value_counts().sort_index()
    for cluster_id, count in train_counts.items():
        print(f"Cluster {cluster_id}: {count} transitions")
    
    print("\nTest cluster distribution:")
    test_counts = test_df['cluster_id'].value_counts().sort_index()
    for cluster_id, count in test_counts.items():
        print(f"Cluster {cluster_id}: {count} transitions")

if __name__ == "__main__":
    input_csv = "../data/data_processing/train_dataset/filtered_dataset.target_frames.clustered.csv"
    train_csv = "../data/data_processing/train_dataset/filtered_dataset.target_frames.clustered.train.16forheatmap.csv"
    test_csv = "../data/data_processing/train_dataset/filtered_dataset.target_frames.clustered.test.16forheatmap.csv"
    samples_per_cluster = 50
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    split_by_clusters(input_csv, train_csv, test_csv, samples_per_cluster)
