import pandas as pd
import numpy as np
from PIL import Image
import ast
from tqdm.auto import tqdm
import os
import torch
from torchvision import transforms
from pathlib import Path

tqdm.pandas()  # Enable progress_apply for pandas

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

def compute_frame_difference(img1_path, img2_path, device='cuda'):
    """Compute MSE between two images using GPU"""
    transform = transforms.ToTensor()
    
    img1 = transform(Image.open(img1_path)).to(device)
    img2 = transform(Image.open(img2_path)).to(device)
    
    with torch.no_grad():
        distance = torch.mean((img2 - img1) ** 2)
        return float(distance.cpu())

def check_sequence(row, cluster_center_path, threshold, device):
    """Check if all images in sequence are within threshold"""
    sequence = ast.literal_eval(row['Image_seq_cond_path']) if isinstance(row['Image_seq_cond_path'], str) else row['Image_seq_cond_path']
    
    for img_path in sequence:
        if compute_frame_difference(cluster_center_path, img_path, device) > threshold:
            return False
    return True

def filter_cluster_sequences(input_csv, cluster_center_path, output_csv, output_dir, 
                           threshold=0.01, device='cuda', history_length=3, debug=False):
    """Filter sequences where all previous images are within threshold distance of cluster center"""
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Reading dataset from {input_csv}")
    df = pd.read_csv(input_csv)
    
    if debug:
        print("Debug mode: using first 1000 rows only")
        df = df.head(1000)
    
    # Filter sequences using pandas
    print("\nFiltering sequences...")
    filtered_df = df[df.progress_apply(lambda row: check_sequence(row, cluster_center_path, threshold, device), axis=1)]
    
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
    debug = False  # Set to True to process only first 1000 rows
    
    filtered_df = filter_cluster_sequences(
        input_csv,
        cluster_center_path,
        output_csv,
        output_dir,
        threshold=threshold,
        device=device,
        history_length=history_length,
        debug=debug
    )
