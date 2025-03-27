import numpy as np
import os
import argparse
from tqdm import tqdm
import json
from glob import glob
import pandas as pd
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Compute mean and std of preprocessed latent vectors")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="computer/train_dataset_encoded",
        help="Directory containing preprocessed .npy files"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="latent_stats.json",
        help="Output file to save mean and std values"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="data/data_processing/train_dataset/train_dataset.target_frames.csv",
        help="CSV file containing record_num and image_num information"
    )
    parser.add_argument(
        "--num_records",
        type=int,
        default=40000,
        help="Number of record folders to sample"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Reading CSV file: {args.csv_file}")

    # Read the CSV file
    df = pd.read_csv(args.csv_file)
    df['record_num'] = df['record_num'].astype(int)
    df['image_num'] = df['image_num'].astype(int)
    
    # Get unique record numbers
    unique_records = df['record_num'].unique()
    print(f"Found {len(unique_records)} unique record folders in CSV")
    
    # Sample a subset of record numbers
    if args.num_records >= len(unique_records):
        sampled_records = unique_records
        print(f"Using all {len(sampled_records)} record folders")
    else:
        sampled_records = np.random.choice(unique_records, size=args.num_records, replace=False)
        print(f"Sampled {len(sampled_records)} record folders out of {len(unique_records)}")

    # Initialize list to store selected .npy files
    selected_npy_files = []
    
    # For each sampled record, randomly select one .npy file
    for record_num in tqdm(sampled_records, desc="Sampling files from records"):
        record_folder = os.path.join(args.data_dir, f"record_{record_num}")
        
        if os.path.exists(record_folder) and os.path.isdir(record_folder):
            # Get all .npy files in this folder
            npy_files = glob(os.path.join(record_folder, "*.npy"))
            
            # Remove padding file if present
            npy_files = [f for f in npy_files if not os.path.basename(f) == "padding.npy"]
            
            if npy_files:
                # Randomly select one file
                selected_file = random.choice(npy_files)
                selected_npy_files.append(selected_file)
    
    print(f"Selected {len(selected_npy_files)} files for processing")
    
    if not selected_npy_files:
        print("Error: No files were selected. Please check your data directory and CSV file.")
        return
    
    # Load a sample file to get dimensions
    sample = np.load(selected_npy_files[0])
    num_channels = sample.shape[0]
    
    print(f"Latent vectors have {num_channels} channels")
    print(f"Sample latent shape: {sample.shape}")
    
    # Initialize arrays to store running sum and sum of squares
    sum_values = np.zeros(num_channels)
    sum_squares = np.zeros(num_channels)
    total_elements = 0
    
    # Process files
    for npy_file in tqdm(selected_npy_files, desc="Processing files"):
        try:
            latent = np.load(npy_file)
            
            # Compute sum per channel
            channel_sum = np.sum(latent, axis=tuple(range(1, latent.ndim)))
            sum_values += channel_sum
            
            # Compute sum of squares per channel
            channel_sum_squares = np.sum(np.square(latent), axis=tuple(range(1, latent.ndim)))
            sum_squares += channel_sum_squares
            
            # Count elements per channel
            n_elements = np.prod(latent.shape[1:])
            total_elements += n_elements
        except Exception as e:
            print(f"Error processing {npy_file}: {e}")
    
    # Calculate mean and std
    mean = sum_values / total_elements
    variance = (sum_squares / total_elements) - np.square(mean)
    std = np.sqrt(variance)
    
    # Convert to Python lists for JSON serialization
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "num_files_processed": len(selected_npy_files),
        "total_elements_per_channel": int(total_elements),
        "sampled_records": sorted(sampled_records.tolist()) if isinstance(sampled_records, np.ndarray) else sorted(sampled_records)
    }
    
    # Save stats to file
    with open(args.output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {args.output_file}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")


if __name__ == "__main__":
    main() 
