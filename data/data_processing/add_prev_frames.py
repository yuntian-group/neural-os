import pandas as pd
import ast
from collections import defaultdict

def convert_to_14_frames(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert string representations of lists to actual lists
    df['Image_seq_cond_path'] = df['Image_seq_cond_path'].apply(ast.literal_eval)
    df['Action_seq'] = df['Action_seq'].apply(ast.literal_eval)
    
    # Create a mapping from image file name to action
    image_to_action = defaultdict(lambda: '0~0')  # Default action if not found
    for _, row in df.iterrows():
        target_image = row['Target_image']
        action = row['Action_seq'][-1]  # The last action in the sequence
        image_to_action[target_image] = action
    
    # Function to get previous image name
    def get_prev_image(image_name):
        parts = image_name.split('_')
        num = int(parts[-1].split('.')[0])
        if num == 0:
            return 'train_dataset/padding.png'
        return f"{parts[0]}_{parts[1]}_{num-1}.png"
    
    # Create new columns for the extended sequences
    df['Extended_Image_seq'] = df['Image_seq_cond_path'].apply(
        lambda seq: [get_prev_image(seq[0]) for _ in range(7)] + seq
    )
    
    df['Extended_Action_seq'] = df.apply(
        lambda row: [image_to_action[img] for img in row['Extended_Image_seq'][:-1]] + [row['Action_seq'][-1]],
        axis=1
    )
    
    # Trim to 14 frames/actions if necessary
    df['Extended_Image_seq'] = df['Extended_Image_seq'].apply(lambda seq: seq[-14:])
    df['Extended_Action_seq'] = df['Extended_Action_seq'].apply(lambda seq: seq[-14:])
    
    # Replace the original columns with the extended ones
    df['Image_seq_cond_path'] = df['Extended_Image_seq']
    df['Action_seq'] = df['Extended_Action_seq']
    
    # Drop the temporary columns
    df = df.drop(columns=['Extended_Image_seq', 'Extended_Action_seq'])
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Converted data saved to {output_file}")

# Usage
input_file = '../../computer/train_dataset/train_dataset.csv'
output_file = '../../computer/train_dataset/train_dataset_14frames.csv'
convert_to_14_frames(input_file, output_file)
