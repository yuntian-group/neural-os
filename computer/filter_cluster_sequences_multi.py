import pandas as pd
import numpy as np
from PIL import Image
import ast
from tqdm.auto import tqdm
from collections import defaultdict
import os
import shutil
from pathlib import Path
import torch
from torchvision import transforms
from multiprocessing import Pool, cpu_count

tqdm.pandas()  # Enable progress_apply for pandas

def compute_frame_difference(img1_path, img2_path, device='cpu'):
    """Compute MSE between two images using GPU"""
    transform = transforms.ToTensor()
    
    img1 = transform(Image.open(img1_path)).to(device)
    img2 = transform(Image.open(img2_path)).to(device)
    
    with torch.no_grad():
        distance = torch.mean((img2 - img1) ** 2)
        return float(distance.cpu())

def check_sequence_parallel(args):
    """Parallel version of check_sequence"""
    sequence, cluster_centers, threshold = args
    
    # For each image in sequence
    for img_path in sequence:
        # Check if image is close to ANY cluster center
        min_distance = float('inf')
        for center_path in cluster_centers:
            distance = compute_frame_difference(center_path, img_path, 'cpu')
            min_distance = min(min_distance, distance)
        
        # If image is not close to any cluster center, reject sequence
        if min_distance > threshold:
            return False
    
    return True

def filter_cluster_sequences_multi(input_csv, cluster_dirs, output_csv, output_dir, 
                                 threshold=0.01, device='cpu', history_length=3, debug=False):
    """Filter sequences where all previous images are within threshold distance of any cluster center"""
    print(f"Reading dataset from {input_csv}")
    df = pd.read_csv(input_csv)
    
    if debug:
        print("Debug mode: using first 1000 rows only")
        df = df.head(1000)
    
    # Collect all cluster center paths
    cluster_centers = []
    for cluster_dir in cluster_dirs:
        center_path = Path(cluster_dir) / "cluster_center.png"
        if center_path.exists():
            cluster_centers.append(str(center_path))
        else:
            print(f"Warning: No cluster center found in {cluster_dir}")
    
    print(f"Found {len(cluster_centers)} cluster centers")
    
    # Prepare sequences for parallel processing
    sequences = [ast.literal_eval(seq) if isinstance(seq, str) else seq 
                for seq in df['Image_seq_cond_path']]
    args = [(seq, cluster_centers, threshold) for seq in sequences]
    
    # Use max(1, cpu_count() - 1) workers by default to leave one core free
    num_workers = max(1, cpu_count() - 1)
    print(f"\nFiltering sequences using {num_workers} CPU workers...")
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(check_sequence_parallel, args),
            total=len(args),
            desc="Filtering"
        ))
    
    # Create filtered dataset
    filtered_df = df[results].copy()
    
    # Save filtered dataset
    filtered_df.to_csv(output_csv, index=False)
    
    print(f"\nFiltering complete!")
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(filtered_df)}")
    print(f"Filtered dataset saved to: {output_csv}")
    
    # Save sample transitions
    save_sample_transitions(filtered_df, output_dir, history_length=history_length)
    
    return filtered_df

if __name__ == "__main__":
    input_csv = "desktop_sequences.csv"
    cluster_dirs = [
        "desktop_transition_clusters/cluster_01_size_1499_desktop_terminal",
        "desktop_transition_clusters/cluster_03_size_1275_desktop_firefox",
        "desktop_transition_clusters/cluster_04_size_799_desktop_root",
        "desktop_transition_clusters/cluster_05_size_738_desktop_trash"
    ]
    output_csv = "desktop_sequences_filtered.csv"
    output_dir = "desktop_transitions_filtered"
    threshold = 0.01
    device = 'cpu'
    history_length = 3  # Number of previous images to show in transition
    debug = False  # Set to True to process only first 1000 rows
    
    filtered_df = filter_cluster_sequences_multi(
        input_csv,
        cluster_dirs,
        output_csv,
        output_dir,
        threshold=threshold,
        device=device,
        history_length=history_length,
        debug=debug
    )
