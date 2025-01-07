import pandas as pd
import numpy as np
from PIL import Image
import ast
from tqdm import tqdm

def compute_frame_difference(current_path, prev_path):
    """Compute MSE between two frames"""
    current_frame = np.array(Image.open(current_path))
    prev_frame = np.array(Image.open(prev_path))
    
    current_norm = current_frame.astype(float) / 255.0
    prev_norm = prev_frame.astype(float) / 255.0
    
    mse = np.mean((current_norm - prev_norm) ** 2)
    return mse

def filter_dataset(input_csv, output_csv, threshold=0.001):
    # Read the CSV file
    print(f"Reading dataset from {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Convert string representations to lists
    df['Image_seq_cond_path'] = df['Image_seq_cond_path'].apply(ast.literal_eval)
    
    # Initialize list to store valid indices
    valid_indices = []
    differences = []
    
    print("Computing frame differences and filtering samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        target_path = row['Target_image']
        prev_frame_path = row['Image_seq_cond_path'][-1]  # Last conditioning frame
        
        # Skip if either path contains 'padding.png'
        if 'padding.png' in prev_frame_path or 'padding.png' in target_path:
            continue
            
        diff = compute_frame_difference(target_path, prev_frame_path)
        
        if diff >= threshold:
            valid_indices.append(idx)
            differences.append(diff)
    
    # Filter the dataframe
    filtered_df = df.iloc[valid_indices].copy()
    
    # Add the difference as a new column
    filtered_df['frame_difference'] = differences
    
    # Save filtered dataset
    filtered_df.to_csv(output_csv, index=False)
    
    print(f"\nFiltering complete!")
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(filtered_df)}")
    print(f"Filtered dataset saved to: {output_csv}")
    
    return filtered_df

if __name__ == "__main__":
    input_csv = "train_dataset/train_dataset_2000.csv"  # Update this path
    output_csv = "train_dataset/filtered_dataset.csv"   # Update this path
    filtered_df = filter_dataset(input_csv, output_csv, threshold=0.001) 