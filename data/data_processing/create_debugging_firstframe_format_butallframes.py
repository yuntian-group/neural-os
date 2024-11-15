import pandas as pd
import ast
from collections import defaultdict

def format_action(action_str, is_padding=False):
    if is_padding:
        return "N N N N N N : N N N N N"
    prefix = 'N'
    items = action_str.split('~')
    if len(items) == 3:
        a = items[0]
        items = items[1:]
        if a == 'left_click':
            prefix = 'L'
        else:
            assert False
    # Split the x~y coordinates
    #x, y = map(int, action_str.split('~'))
    x, y = items
    x = int(x)
    y = int(y)
    
    # Convert numbers to padded strings and add spaces between digits
    x_str = f"{abs(x):04d}"
    y_str = f"{abs(y):04d}"
    x_spaced = ' '.join(x_str)
    y_spaced = ' '.join(y_str)
    
    # Format with sign and proper spacing
    return prefix + ' ' + f"{'+ ' if x >= 0 else '- '}{x_spaced} : {'+ ' if y >= 0 else '- '}{y_spaced}"

def filter_first_frames(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Filter rows where the target image ends with 'image_0.png'
    first_frame_df = df #df[df['Target_image'].str.endswith('image_0.png')]
    
    # Convert string representations of lists to actual lists
    first_frame_df['Image_seq_cond_path'] = first_frame_df['Image_seq_cond_path'].apply(ast.literal_eval)
    first_frame_df['Action_seq'] = first_frame_df['Action_seq'].apply(ast.literal_eval)
    
    # Format each action in the Action_seq
    first_frame_df['Action_seq'] = first_frame_df.apply(
        lambda row: [format_action(action, 'padding.png' in img) 
                    for action, img in zip(row['Action_seq'], row['Image_seq_cond_path'] + [row['Target_image']])],
        axis=1
    )
    
    # Convert back to string representation
    first_frame_df['Action_seq'] = first_frame_df['Action_seq'].apply(str)
    first_frame_df['Image_seq_cond_path'] = first_frame_df['Image_seq_cond_path'].apply(str)
    
    # Repeat each row 100 times
    #repeated_df = pd.concat([first_frame_df] * 100, ignore_index=True)
    repeated_df = pd.concat([first_frame_df] * 1, ignore_index=True)
    
    # Save the filtered DataFrame to a new CSV file
    repeated_df.to_csv(output_file, index=False)
    print(f"First frame data saved to {output_file}")

# Usage
input_file = '../../computer/train_dataset/train_dataset_14frames.csv'
output_file = '../../computer/train_dataset/train_dataset_14frames_firstframe_allframes.csv'
filter_first_frames(input_file, output_file)
