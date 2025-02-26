import pandas as pd
import numpy as np
from PIL import Image
import ast
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count

def compute_frame_difference(image):
    """Compute MSE between two frames"""
    record_num, image_num = image

    current_path = f"./data/data_processing/train_dataset/record_{record_num}/image_{image_num}.png"
    if image_num == 0:
        prev_path = f"./data/data_processing/train_dataset/record_0/image_0.png"
    else:
        prev_path = f"./data/data_processing/train_dataset/record_{record_num}/image_{image_num - 1}.png"

    # compute distance from current frame to desktop
    desktop_path = f"./data/data_processing/train_dataset/record_0/image_0.png"
    if 'padding.png' in prev_path:
        distance_prev_to_desktop = -1
    else:
        distance_prev_to_desktop = compute_distance(prev_path, desktop_path)

    # compute distance between prev and current frame
    if 'padding.png' in prev_path or 'padding.png' in current_path:
        distance_current_to_prev = -1
    else:
        distance_current_to_prev = compute_distance(current_path, prev_path)

    return (distance_prev_to_desktop, distance_current_to_prev)

def compute_distance(current_path, prev_path):
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
    
    # Convert string representations to integers
    df['image_num'] = df['image_num'].apply(int)
    df['record_num'] = df['record_num'].apply(int)
    
    # Prepare images for parallel processing
    images = [(row['record_num'], row['image_num']) 
                  for _, row in df.iterrows()]
    
    # Use max(1, cpu_count() - 1) workers by default
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
        num_workers = min(num_workers, 64)
    
    print(f"Computing frame differences using {num_workers} workers...")
    with Pool(num_workers) as pool:
        differences = list(tqdm(
            pool.imap(compute_frame_difference, images),
            total=len(images)
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
        df['frame_difference'] = df['frame_difference'].apply(ast.literal_eval)
    
    # Step 2: Filter based on threshold
    def filter_frame_difference(row):
        return row['frame_difference'][1] >= threshold
    
    print("\nFiltering dataset...")
    filtered_df = df[df.apply(filter_frame_difference, axis=1)]
    # Save filtered dataset
    filtered_df.to_csv(output_csv, index=False)
    
    print(f"\nFiltering complete!")
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(filtered_df)}")
    print(f"Filtered dataset saved to: {output_csv}")
    
    return filtered_df

if __name__ == "__main__":
    input_csv = "computer/train_dataset/train_dataset.target_frames.csv"  # Update this path
    distances_csv = "computer/train_dataset/dataset_with_distances.csv"  # Intermediate file with distances
    output_csv = "computer/train_dataset/filtered_dataset.target_frames.csv"   # Final filtered dataset
    
    # By default, it will use (CPU count - 1) workers
    # You can specify a different number of workers if desired
    filtered_df = filter_dataset(
        input_csv, 
        output_csv, 
        distances_csv, 
        threshold=0.1, # old value 0.001 
        force_recompute=False,
        num_workers=None# Set to a specific number (e.g., 4) if desired
    )
