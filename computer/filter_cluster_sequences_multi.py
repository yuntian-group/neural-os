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
    sequence, target_image, cluster_centers, threshold = args
    desktop_center_path = 'clustering_results/cluster_01_size_2728_desktop/cluster_center.png'
    
    # For each cluster center
    for center_path in cluster_centers:
        # Check if ALL images in sequence AND target are close to THIS cluster center
        sequence_ok = True
        
        # Check all conditional images
        for img_path in sequence:
            if compute_frame_difference(desktop_center_path, img_path, 'cpu') > threshold:
                sequence_ok = False
                break
        
        # Check target image
        if sequence_ok and compute_frame_difference(center_path, target_image, 'cpu') > threshold:
            sequence_ok = False
        
        # If all images are close to this cluster center, accept sequence
        if sequence_ok:
            return True
    
    # If no cluster center matches all images, reject sequence
    return False

def filter_cluster_sequences_multi(input_csv, cluster_dirs, output_csv, output_dir, 
                                 threshold=0.01, device='cpu', history_length=3, debug=False,
                                 load_existing=True):
    """Filter sequences where all images (conditional and target) are within threshold distance of the same cluster center"""
    
    if load_existing and os.path.exists(output_csv):
        print(f"Loading existing filtered dataset from {output_csv}")
        filtered_df = pd.read_csv(output_csv)
        print(f"Loaded {len(filtered_df)} sequences")
        
        # Save sample transitions
        save_sample_transitions(filtered_df, output_dir, history_length=history_length)
        
        return filtered_df
    
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
    target_images = df['Target_image']
    args = [(seq, target, cluster_centers, threshold) 
            for seq, target in zip(sequences, target_images)]
    
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

def create_transition_image(sequence_paths, target_path, history_length=3):
    """Create a horizontal strip of images showing the transition"""
    # Take last history_length images from sequence
    sequence_paths = sequence_paths[-history_length:]
    
    # Load all images
    images = [Image.open(path) for path in sequence_paths]
    images.append(Image.open(target_path))  # Add target image
    
    # Get dimensions
    width = images[0].width
    height = images[0].height
    
    # Create new image
    total_width = width * len(images)
    combined_image = Image.new('RGB', (total_width, height))
    
    # Paste images horizontally
    for i, img in enumerate(images):
        combined_image.paste(img, (i * width, 0))
    
    return combined_image

def save_sample_transitions(filtered_df, output_dir, num_samples=20, history_length=3, seed=42):
    """Save sample transitions as horizontal image strips"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample random sequences
    np.random.seed(seed)
    sample_indices = np.random.choice(len(filtered_df), min(num_samples, len(filtered_df)), replace=False)
    
    print(f"\nSaving {len(sample_indices)} sample transitions...")
    for i, idx in enumerate(sample_indices):
        row = filtered_df.iloc[idx]
        sequence_paths = ast.literal_eval(row['Image_seq_cond_path']) if isinstance(row['Image_seq_cond_path'], str) else row['Image_seq_cond_path']
        target_path = row['Target_image']
        
        # Create and save transition image
        transition_image = create_transition_image(sequence_paths, target_path, history_length)
        transition_image.save(output_dir / f"transition_{i:03d}.png")

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
    debug = True # Set to True to process only first 1000 rows
    load_existing = False # Set to True to load from existing CSV
    
    filtered_df = filter_cluster_sequences_multi(
        input_csv,
        cluster_dirs,
        output_csv,
        output_dir,
        threshold=threshold,
        device=device,
        history_length=history_length,
        debug=debug,
        load_existing=load_existing
    )
