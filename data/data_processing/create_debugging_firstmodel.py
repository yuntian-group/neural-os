import pandas as pd
import ast
from collections import defaultdict

def repeat_first_video(input_file, output_file, repeat_count=1000):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Filter rows where the target image belongs to the first video (record_0)
    first_video_df = df[df['Target_image'].str.contains('record_0')]
    
    # Repeat the first video data
    repeated_df = pd.concat([first_video_df] * repeat_count, ignore_index=True)
    
    # Save the repeated DataFrame to a new CSV file
    repeated_df.to_csv(output_file, index=False)
    print(f"Repeated data saved to {output_file}")

# Usage
input_file = '../../computer/train_dataset/train_dataset_14frames.csv'
output_file = '../../computer/train_dataset/train_dataset_debugging.csv'
repeat_first_video(input_file, output_file) 