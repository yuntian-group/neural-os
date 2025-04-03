#!/usr/bin/env python3
import pandas as pd
import os
import shutil
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Copy encoded record folders for transfer")
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/data_processing/train_dataset/filtered_dataset.target_frames.clustered.train_shuffled.shuffled.csv",
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/data_processing/train_dataset/filtered_dataset.target_frames.clustered.train.head100.csv",
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--encoded_dir",
        type=str,
        default="computer/train_dataset_encoded",
        help="Directory containing encoded record folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="computer/train_dataset_encoded/to_transfer_train_head100",
        help="Directory to copy record folders to"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print actions without copying files"
    )
    return parser.parse_args()

def get_record_numbers(csv_files):
    """Extract unique record numbers from CSV files"""
    record_nums = set()
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file {csv_file} not found, skipping")
            continue
            
        print(f"Reading record numbers from {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            if 'record_num' in df.columns:
                # Add all unique record numbers to the set
                record_nums.update(df['record_num'].unique())
            else:
                print(f"Warning: 'record_num' column not found in {csv_file}")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return sorted(list(record_nums))

def copy_record_folders(record_nums, encoded_dir, output_dir, dry_run=False):
    """Copy record folders to output directory"""
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
    
    total_size = 0
    for record_num in tqdm(record_nums, desc="Copying record folders"):
        src_dir = os.path.join(encoded_dir, f"record_{record_num}")
        dst_dir = os.path.join(output_dir, f"record_{record_num}")
        
        if not os.path.exists(src_dir):
            print(f"Warning: Source directory {src_dir} not found, skipping")
            continue
        
        # Calculate directory size for reporting
        dir_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                      for dirpath, _, filenames in os.walk(src_dir)
                      for filename in filenames)
        total_size += dir_size
        
        if dry_run:
            print(f"Would copy {src_dir} to {dst_dir} (size: {dir_size / (1024*1024):.2f} MB)")
        else:
            try:
                if os.path.exists(dst_dir):
                    print(f"Warning: Destination directory {dst_dir} already exists, skipping")
                    continue
                    
                shutil.copytree(src_dir, dst_dir)
                print(f"Copied {src_dir} to {dst_dir} (size: {dir_size / (1024*1024):.2f} MB)")
            except Exception as e:
                print(f"Error copying {src_dir}: {e}")
    
    print(f"\nTotal size of copied data: {total_size / (1024*1024*1024):.2f} GB")
    return total_size

def main():
    args = parse_args()
    
    # Get unique record numbers from CSV files
    #csv_files = [args.train_csv, args.test_csv]
    csv_files = [args.test_csv]
    record_nums = get_record_numbers(csv_files)
    
    print(f"Found {len(record_nums)} unique record numbers")
    
    if len(record_nums) == 0:
        print("No record numbers found, exiting")
        return
    
    # Copy record folders to output directory
    total_size = copy_record_folders(record_nums, args.encoded_dir, args.output_dir, args.dry_run)
    
    if not args.dry_run:
        print(f"\nSuccessfully copied {len(record_nums)} record folders to {args.output_dir}")
        print(f"To compress the folder for transfer, you can use:")
        print(f"  tar -czf to_transfer.tar.gz {args.output_dir}/")
        print(f"  # or")
        print(f"  zip -r to_transfer.zip {args.output_dir}/")

if __name__ == "__main__":
    main() 
