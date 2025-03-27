#!/usr/bin/env python3
import pandas as pd
import os
import time

def main():
    # File paths
    clustered_test_csv = "../data/data_processing/train_dataset/filtered_dataset.target_frames.clustered.test.csv"
    
    # Source CSV files
    filtered_csv = "../data/data_processing/train_dataset/filtered_dataset.target_frames.csv"
    train_csv = "../data/data_processing/train_dataset/train_dataset.target_frames.csv"
    
    # Step 1: Extract record_nums from the clustered test CSV
    print(f"Reading test record numbers from {clustered_test_csv}")
    try:
        test_df = pd.read_csv(clustered_test_csv)
        test_record_nums = set(test_df['record_num'].unique())
        print(f"Found {len(test_record_nums)} unique record numbers in the test set")
    except Exception as e:
        print(f"Error reading {clustered_test_csv}: {e}")
        return
    
    # For both source files, we'll split them based on the test record numbers
    for source_file in [filtered_csv, train_csv]:
        base_name = os.path.basename(source_file)
        base_name_without_ext = os.path.splitext(base_name)[0]
        output_dir = os.path.dirname(source_file)
        
        train_output = os.path.join(output_dir, f"{base_name_without_ext}.train.csv")
        test_output = os.path.join(output_dir, f"{base_name_without_ext}.test.csv")
        
        print(f"\nProcessing {source_file}")
        try:
            start_time = time.time()
            df = pd.read_csv(source_file)
            
            # Check if record_num column exists
            if 'record_num' not in df.columns:
                print(f"Error: 'record_num' column not found in {source_file}")
                continue
            
            # Split the dataframe
            test_rows = df[df['record_num'].isin(test_record_nums)]
            train_rows = df[~df['record_num'].isin(test_record_nums)]
            
            # Save the files
            test_rows.to_csv(test_output, index=False)
            train_rows.to_csv(train_output, index=False)
            
            elapsed = time.time() - start_time
            print(f"Split {len(df)} rows into {len(train_rows)} train and {len(test_rows)} test rows")
            print(f"Saved train data to {train_output}")
            print(f"Saved test data to {test_output}")
            print(f"Processing completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing {source_file}: {e}")
    
    print("\nAll processing completed!")

if __name__ == "__main__":
    main()
