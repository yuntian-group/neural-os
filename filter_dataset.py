import pandas as pd
import numpy as np
from PIL import Image
import ast
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count

def compute_frame_difference(image_pair):
    """Compute MSE between two frames"""
    current_path, prev_path = image_pair
    
    # Handle padding frames
    if 'padding.png' in prev_path or 'padding.png' in current_path:
        return -1
        
    current_frame = np.array(Image.open(current_path))
    prev_frame = np.array(Image.open(prev_path))
    
    current_norm = current_frame.astype(float) / 255.0
    prev_norm = prev_frame.astype(float) / 255.0
    
    mse = np.mean((current_norm - prev_norm) ** 2)
    return mse

def compute_dataset_distances(input_csv, distances_csv, num_workers=None):
    """First step: compute and save frame differences using multiple processes"""
    print(f"Reading dataset from {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Convert string representations to lists
    df['Image_seq_cond_path'] = df['Image_seq_cond_path'].apply(ast.literal_eval)
    
    # Prepare image pairs for parallel processing
    image_pairs = [(row['Target_image'], row['Image_seq_cond_path'][-1]) 
                  for _, row in df.iterrows()]
    
    # Use max(1, cpu_count() - 1) workers by default
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Computing frame differences using {num_workers} workers...")
    with Pool(num_workers) as pool:
        differences = list(tqdm(
            pool.imap(compute_frame_difference, image_pairs),
            total=len(image_pairs)
        ))
    
    # Add the difference as a new column
    df['frame_difference'] = differences
    
    # Save dataset with distances
    df.to_csv(distances_csv, index=False)
    print(f"Dataset with distances saved to: {distances_csv}")
    
    return df

def filter_dataset(input_csv, output_csv, distances_csv, threshold=0.001, force_recompute=False, num_workers=None):
    """Main function that handles both steps"""
    # Step 1: Get or compute distances
    if not os.path.exists(distances_csv) or force_recompute:
        print("Computing frame differences (this may take a while)...")
        df = compute_dataset_distances(input_csv, distances_csv, num_workers)
    else:
        print(f"Loading pre-computed distances from {distances_csv}")
        df = pd.read_csv(distances_csv)
    
    # Step 2: Filter based on threshold
    print("\nFiltering dataset...")
    filtered_df = df[df['frame_difference'] >= threshold].copy()
    
    # Save filtered dataset
    filtered_df.to_csv(output_csv, index=False)
    
    print(f"\nFiltering complete!")
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(filtered_df)}")
    print(f"Filtered dataset saved to: {output_csv}")
    
    return filtered_df

if __name__ == "__main__":
    input_csv = "train_dataset/train_dataset.csv"  # Update this path
    distances_csv = "train_dataset/dataset_with_distances.csv"  # Intermediate file with distances
    output_csv = "train_dataset/filtered_dataset.csv"   # Final filtered dataset
    
    # By default, it will use (CPU count - 1) workers
    # You can specify a different number of workers if desired
    filtered_df = filter_dataset(
        input_csv, 
        output_csv, 
        distances_csv, 
        threshold=0.001, 
        force_recompute=True,
        num_workers=16# Set to a specific number (e.g., 4) if desired
    ) 
