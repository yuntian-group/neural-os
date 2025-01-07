import pandas as pd
import numpy as np
from PIL import Image
import ast
from tqdm import tqdm
import os

def compute_frame_difference(current_path, prev_path):
    """Compute MSE between two frames"""
    current_frame = np.array(Image.open(current_path))
    prev_frame = np.array(Image.open(prev_path))
    
    current_norm = current_frame.astype(float) / 255.0
    prev_norm = prev_frame.astype(float) / 255.0
    
    mse = np.mean((current_norm - prev_norm) ** 2)
    return mse

def compute_dataset_distances(input_csv, distances_csv):
    """First step: compute and save frame differences"""
    print(f"Reading dataset from {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Convert string representations to lists
    df['Image_seq_cond_path'] = df['Image_seq_cond_path'].apply(ast.literal_eval)
    
    # Initialize list to store differences
    differences = []
    
    print("Computing frame differences...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        target_path = row['Target_image']
        prev_frame_path = row['Image_seq_cond_path'][-1]  # Last conditioning frame
        
        if 'padding.png' in prev_frame_path or 'padding.png' in target_path:
            diff = -1  # Use -1 to mark padding frames
        else:
            diff = compute_frame_difference(target_path, prev_frame_path)
        
        differences.append(diff)
    
    # Add the difference as a new column
    df['frame_difference'] = differences
    
    # Save dataset with distances
    df.to_csv(distances_csv, index=False)
    print(f"Dataset with distances saved to: {distances_csv}")
    
    return df

def filter_dataset(input_csv, output_csv, distances_csv, threshold=0.001, force_recompute=False):
    """Main function that handles both steps"""
    # Step 1: Get or compute distances
    if not os.path.exists(distances_csv) or force_recompute:
        print("Computing frame differences (this may take a while)...")
        df = compute_dataset_distances(input_csv, distances_csv)
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
    input_csv = "train_dataset/train_dataset_2000.csv"  # Update this path
    distances_csv = "train_dataset/dataset_with_distances.csv"  # Intermediate file with distances
    output_csv = "train_dataset/filtered_dataset.csv"   # Final filtered dataset
    
    # Set force_recompute=True if you want to recompute distances even if the file exists
    filtered_df = filter_dataset(input_csv, output_csv, distances_csv, 
                               threshold=0.001, force_recompute=False) 