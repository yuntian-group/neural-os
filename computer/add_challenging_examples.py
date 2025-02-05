import pandas as pd
import re
import numpy as np
import pickle
offset = 25
input_file = 'desktop_sequences_filtered_with_desktop_1.5k_removelast100.csv'
input_file = 'desktop_sequences_filtered_with_desktop_1.5k_last100.csv'
output_file = input_file[:-4] + '.challenging.csv'
# Load the action mapping dictionary
with open('image_action_mapping.pkl', 'rb') as f:
    mapping_dict = pickle.load(f)

# Read the CSV file
df = pd.read_csv(input_file)

# Convert string representations of lists to actual lists
df['Image_seq_cond_path'] = df['Image_seq_cond_path'].apply(eval)
df['Action_seq'] = df['Action_seq'].apply(eval)

# Remove rows containing desktop_desktop
df = df[~df['matched_cluster'].str.contains('desktop_desktop')]

# Function to create new row for firefox cases
def create_firefox_row(row):
    # Extract record number using regex
    match = re.search(r'record_(\d+)', row['Image_seq_cond_path'][0])
    if not match:
        return None
    
    record_num = int(match.group(1))
    
    # Create new paths
    #padding_paths = ['train_dataset/padding.png'] * 7
    image_paths = [f'train_dataset/record_{record_num}/image_{i}.png' for i in range(offset-7, offset+7)]
    new_seq_path = image_paths
    
    # Generate new action sequence using mapping_dict
    new_action_seq = []
    for img_path in new_seq_path + [f'train_dataset/record_{record_num}/image_{offset+7}.png']:  # include target image
        if 'padding.png' in img_path:
            new_action_seq.append('N + 0 0 0 0 : + 0 0 0 0')  # dummy action for padding
        else:
            # Extract image number from path
            img_match = re.search(r'image_(\d+)', img_path)
            if img_match:
                img_num = int(img_match.group(1))
                new_action_seq.append(mapping_dict.get((record_num, img_num)))
    
    # Create new row
    return {
        'Unnamed: 0': row['Unnamed: 0'],
        'Image_seq_cond_path': new_seq_path,
        'Action_seq': new_action_seq,
        'Target_image': f'train_dataset/record_{record_num}/image_{offset+7}.png',
        'frame_difference': row['frame_difference'],
        'matched_cluster': 'None'
    }

# Create new rows for firefox cases
firefox_rows = []
for idx, row in df[df['matched_cluster'].str.contains('desktop_firefox')].iterrows():
    new_row = create_firefox_row(row)
    if new_row is not None:
        firefox_rows.append(new_row)

# Add new rows to dataframe
df_new = pd.concat([df] + [pd.DataFrame([row]) for row in firefox_rows], ignore_index=True)

# Shuffle the dataframe
df_new = df_new.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the processed dataframe
df_new.to_csv(output_file, index=False)
