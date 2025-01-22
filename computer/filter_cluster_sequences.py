import pandas as pd
import numpy as np
from PIL import Image
import ast
from tqdm import tqdm
import os
import torch
from torchvision import transforms
from pathlib import Path

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

def filter_cluster_sequences(input_csv, cluster_center_path, output_csv, output_dir, 
                           threshold=0.01, device='cuda', history_length=3):
    """Filter sequences where all previous images are within threshold distance of cluster center"""
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Reading dataset from {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Convert string representations to lists
    df['Image_seq_cond_path'] = df['Image_seq_cond_path'].apply(ast.literal_eval)
    
    # Prepare all images for comparison
    all_seq_images = []
    seq_to_row_map = {}  # Map to track which sequences each image belongs to
    
    for idx, row in df.iterrows():
        for img_path in row['Image_seq_cond_path']:
            all_seq_images.append(img_path)
            seq_to_row_map[img_path] = idx
    
    # Compute distances to cluster center
    print("\nComputing distances to cluster center...")
    distances = compute_frame_difference_batch(
        [cluster_center_path] * len(all_seq_images),
        all_seq_images,
        device=device
    )
    
    # Create distance lookup dictionary
    distance_lookup = {img: dist for img, dist in zip(all_seq_images, distances)}
    
    # Filter sequences
    print("\nFiltering sequences...")
    keep_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering"):
        # Check if all images in sequence are within threshold
        sequence_ok = all(
            distance_lookup[img_path] <= threshold 
            for img_path in row['Image_seq_cond_path']
        )
        if sequence_ok:
            keep_rows.append(idx)
    
    # Create filtered dataset
    filtered_df = df.loc[keep_rows].copy()
    
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
    input_csv = "train_dataset/filtered_dataset.csv"
    cluster_center_path = "clustering_results/cluster_01_size_2728_desktop/cluster_center.png"
    output_csv = "train_dataset/desktop_sequences.csv"
    output_dir = "train_dataset/desktop_transitions"
    threshold = 0.01
    device = 'cuda'
    history_length = 3  # Number of previous images to show in transition
    
    filtered_df = filter_cluster_sequences(
        input_csv,
        cluster_center_path,
        output_csv,
        output_dir,
        threshold=threshold,
        device=device,
        history_length=history_length
    )
