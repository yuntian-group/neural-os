import pandas as pd
import re
import numpy as np
import pickle
from tqdm import tqdm

offset = 25

input_files = ['train_dataset_encoded/filtered_dataset.target_frames.train.csv',
               'train_dataset_encoded2/train_dataset_apr3_encoded/filtered_dataset.target_frames.csv',
               'train_dataset_encoded3/train_dataset_apr2_encoded/filtered_dataset.target_frames.csv',
               'train_dataset_encoded4/train_dataset_apr5_encoded/filtered_dataset.target_frames.csv',
               'train_dataset_encoded5/train_dataset_apr14_2_encoded/filtered_dataset.target_frames.csv']

def create_new_row(record_num, image_num):
    if offset + 7 >= image_num:
        return None
    return {
        'record_num': record_num,
        'image_num': offset + 7,
        'frame_difference': "(-1, -1)"
    }

print(f"Processing {len(input_files)} files...")
for input_file in tqdm(input_files, desc="Processing files"):
    output_file = input_file[:-4] + '.challenging.csv'
    
    df = pd.read_csv(input_file)
    
    # Create all new rows at once using list comprehension
    new_rows = [create_new_row(row['record_num'], row['image_num']) 
                for _, row in tqdm(df.iterrows(), desc=f"Processing {input_file}", total=len(df))]
    new_rows = [row for row in new_rows if row is not None]
    
    # Create new dataframe directly from the list of dictionaries
    df_new = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df_new.to_csv(output_file, index=False)
