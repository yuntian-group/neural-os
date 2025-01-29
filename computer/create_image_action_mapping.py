import pandas as pd
import re
import ast
from tqdm import tqdm
import pickle

def extract_numbers(path):
    """Extract record and image numbers from path"""
    if 'padding.png' in path:
        return -1, -1
    match = re.search(r'record_(\d+)/image_(\d+)', path)
    if match:
        return int(match.group(1)), int(match.group(2))
    assert False, path
    return None

def create_mapping():
    # Read the original dataset
    print("Reading dataset...")
    df = pd.read_csv('train_dataset/train_dataset.csv')
    
    # Create mapping dictionary
    mapping_dict = {}  # Using (record_num, image_num) as key
    
    print("Processing sequences...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating mapping"):
        # Get image sequence and actions
        image_seq = ast.literal_eval(row['Image_seq_cond_path'])
        actions = ast.literal_eval(row['Action_seq'])
        target_img = row['Target_image']
        
        # Process each image and its corresponding action
        for img, action in zip(image_seq, actions[:-1]):  # All but last action
            numbers = extract_numbers(img)
            if numbers is not None:  # not a padding image
                mapping_dict[numbers] = action
        
        # Handle target image and last action
        numbers = extract_numbers(target_img)
        if numbers is not None:  # not a padding image
            mapping_dict[numbers] = actions[-1]
    
    # Save dictionary using pickle
    print(f"Saving mapping with {len(mapping_dict)} entries...")
    with open('image_action_mapping.pkl', 'wb') as f:
        pickle.dump(mapping_dict, f)
    print("Done!")


if __name__ == "__main__":
    create_mapping()
