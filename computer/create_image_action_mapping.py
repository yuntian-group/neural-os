import pandas as pd
import re
import ast
from tqdm import tqdm

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
    mapping_data = []
    
    # Add padding action
    PADDING_ACTION = 'N N N N N N : N N N N N'
    
    print("Processing sequences...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating mapping"):
        # Get image sequence and actions
        image_seq = ast.literal_eval(row['Image_seq_cond_path'])
        actions = ast.literal_eval(row['Action_seq'])
        target_img = row['Target_image']
        
        # Process each image and its corresponding action
        for img, action in zip(image_seq, actions[:-1]):  # All but last action
            numbers = extract_numbers(img)
            if numbers is None:  # padding image
                continue  # Skip padding images as they'll be handled during lookup
            record_num, img_num = numbers
            mapping_data.append({
                'record_num': record_num,
                'image_num': img_num,
                'action': action
            })
        
        # Handle target image and last action
        numbers = extract_numbers(target_img)
        if numbers is not None:  # not a padding image
            record_num, img_num = numbers
            mapping_data.append({
                'record_num': record_num,
                'image_num': img_num,
                'action': actions[-1]
            })
    
    # Create DataFrame and save
    print("Creating DataFrame and removing duplicates...")
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df = mapping_df.drop_duplicates()  # Remove any duplicates
    mapping_df.to_csv('image_action_mapping.csv', index=False)
    print(f"Created mapping with {len(mapping_df)} entries")


def get_actions_for_sequence(mapping_df, record_num, image_nums):
    """Get actions for a sequence of frames"""
    PADDING_ACTION = 'N N N N N N : N N N N N'
    actions = []
    for img_num in image_nums:
        if img_num < 0:  # Handle negative frame numbers
            actions.append(PADDING_ACTION)
            continue
            
        matches = mapping_df[
            (mapping_df['record_num'] == record_num) & 
            (mapping_df['image_num'] == img_num)
        ]
        if len(matches) == 0:  # No matching action found
            actions.append(PADDING_ACTION)
        else:
            actions.append(matches['action'].iloc[0])
    return actions

if __name__ == "__main__":
    create_mapping()
