import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re  # Add to imports at top
from multiprocessing import Pool
import functools
import os
def compute_distance_to_centers(row, cluster_centers, cluster_ids, transform, device='cpu'):
    """Compute distance between a transition and all cluster centers"""
    # Load the transition images
    record_num = row['record_num']
    image_num = row['image_num']
    
    if image_num == 0:
        prev_path = '../data/data_processing/train_dataset/padding.png'
    else:
        prev_path = f'../data/data_processing/train_dataset/record_{record_num}/image_{image_num - 1}.png'
    curr_path = f'../data/data_processing/train_dataset/record_{record_num}/image_{image_num}.png'
    
    # Load and transform images
    prev_img = transform(Image.open(prev_path)).view(-1)
    curr_img = transform(Image.open(curr_path)).view(-1)
    transition = torch.cat([prev_img, curr_img], dim=0).to(device)
    
    # Compute distances to all centers
    with torch.no_grad():
        # Stack all centers into a single tensor [num_centers, feature_dim]
        centers = torch.stack(cluster_centers)
        
        # Compute distances in one go using broadcasting
        # transition shape: [feature_dim]
        # centers shape: [num_centers, feature_dim]
        distances = torch.norm(transition.unsqueeze(0) - centers, dim=1) ** 2 / transition.size(0)
        distances = distances.cpu()
        min_idx = distances.argmin().item()
        return distances[min_idx].item(), cluster_ids[min_idx]

def process_row(row_data, cluster_centers, cluster_ids, transform, threshold, device='cpu'):
    """Process a single row (to be used with multiprocessing)"""
    try:
        row = pd.Series(row_data)
        min_dist, cluster_id = compute_distance_to_centers(row, cluster_centers, cluster_ids, transform, device)
        if min_dist <= threshold:
            return {
                **row_data,
                'cluster_id': cluster_id,
                'distance_to_center': min_dist
            }
    except Exception as e:
        print(f"Error processing row {row['record_num']}, {row['image_num']}: {e}")
    return None

def filter_by_clusters(input_csv, cluster_dir, output_csv, threshold=0.01, device='cpu', num_workers=8):
    # Load the dataset
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")
    
    # Load cluster centers
    cluster_centers = []
    cluster_ids = []
    cluster_paths = sorted(Path(cluster_dir).glob("cluster_*_size_*"))
    transform = transforms.ToTensor()
    
    print("Loading cluster centers...")
    for cluster_path in cluster_paths:
        if "noise" in str(cluster_path):
            continue
            
        # Extract cluster ID using regex (e.g., "cluster_5_size_100" -> 5)
        match = re.search(r'cluster_(\d+)_size_', str(cluster_path.name))
        if not match:
            continue
        cluster_id = int(match.group(1))
        
        # Find center images
        center_files = list(cluster_path.glob("cluster_center_*.png"))
        if not center_files:
            continue
            
        # Load and concatenate center images
        prev_img = transform(Image.open([f for f in center_files if 'prev' in str(f)][0])).view(-1)
        curr_img = transform(Image.open([f for f in center_files if 'curr' in str(f)][0])).view(-1)
        center = torch.cat([prev_img, curr_img], dim=0).to(device)
        
        cluster_centers.append(center)
        cluster_ids.append(cluster_id)
    
    print(f"Loaded {len(cluster_centers)} cluster centers")
    
    # Prepare the worker function with fixed arguments
    process_row_partial = functools.partial(
        process_row,
        cluster_centers=cluster_centers,
        cluster_ids=cluster_ids,
        transform=transform,
        threshold=threshold,
        device=device
    )
    
    # Convert DataFrame rows to dictionaries for multiprocessing
    row_dicts = df.to_dict('records')
    
    # Process rows in parallel
    print("Computing distances and filtering...")
    #with Pool(num_workers) as pool:
    #    results = list(tqdm(
    #        pool.imap(process_row_partial, row_dicts, chunksize=100),
    #        total=len(row_dicts)
    #    ))
    results = []
    for row in tqdm(row_dicts):
        result = process_row_partial(row)
        if result is not None:
            results.append(result)
    
    # Filter out None results and create DataFrame
    filtered_df = pd.DataFrame(results)
    print(f"Filtered to {len(filtered_df)} rows")
    
    # Save results
    filtered_df.to_csv(output_csv, index=False)
    print(f"Saved filtered dataset to {output_csv}")
    
    # Print statistics
    print("\nStatistics:")
    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(filtered_df)}")
    print(f"Reduction: {(1 - len(filtered_df)/len(df))*100:.1f}%")
    print("\nCluster distribution:")
    cluster_counts = filtered_df['cluster_id'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} transitions")

if __name__ == "__main__":
    input_csv = "train_dataset/filtered_dataset.target_frames.csv"
    cluster_dir = "filtered_transition_clusters"
    output_csv = "train_dataset/filtered_dataset.target_frames.clustered.csv"
    threshold = 0.01
    device = 'cpu'
    # use cpu cores as num_workers
    num_workers = os.cpu_count()
    
    filter_by_clusters(input_csv, cluster_dir, output_csv, threshold, device, num_workers)
