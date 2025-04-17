import pandas as pd
import re
import numpy as np
import pickle
offset = 25



input_files = ['train_dataset_encoded/filtered_dataset.target_frames.train.csv',
               'train_dataset_encoded2/train_dataset_apr3_encoded/filtered_dataset.target_frames.csv',
               'train_dataset_encoded3/train_dataset_apr2_encoded/filtered_dataset.target_frames.csv',
               'train_dataset_encoded4/train_dataset_apr5_encoded/filtered_dataset.target_frames.csv',
               'train_dataset_encoded5/train_dataset_apr14_2_encoded/filtered_dataset.target_frames.csv']


def create_new_row(row):
    # Extract record number using regex
    record_num = row['record_num']
    image_num = row['image_num']
    return {
        'record_num': record_num,
        'image_num': offset + 7,
    }

for input_file in input_files:
    output_file = input_file[:-4] + '.challenging.csv'
    
    df = pd.read_csv(input_file)
    new_rows = []
    for idx, row in df.iterrows():
        new_row = create_row(row)
        if new_row is not None:
            new_rows.append(new_row)
    
    df_new = pd.concat([df] + [pd.DataFrame([row]) for row in new_rows], ignore_index=True)
    df_new = df_new.reset_index(drop=True)
    df_new.to_csv(output_file, index=False)