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
    df = pd.read_csv('../data/data_processing/train_dataset/train_dataset.csv')
    
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
    
    # mapping_dict has key_events like [('keydown', 'space'), ('keydown', 'esc')], but we want to convert it to [('down', 'space'), ('down', 'esc')] by looking up all frames up to this point to check if the key is down or up
    mapping_dict_with_key_states = {}
    for (record_num, image_num), action in tqdm(mapping_dict.items(), desc="Creating mapping with key states"):
        #import pdb; pdb.set_trace()
        if record_num == -1:
            assert image_num == -1
            x, y, left_click, right_click, key_events = action
            mapping_dict_with_key_states[(record_num, image_num)] = (x, y, left_click, right_click, [])
            continue
        assert record_num >= 0 and image_num >= 0, (record_num, image_num)
        down_keys = set([])
        for image_num_prev in range(image_num+1):
            action_prev = mapping_dict[(record_num, image_num_prev)]
            key_events = action_prev[-1]
            for key_state, key in key_events:
                if key_state == 'keydown':
                    down_keys.add(key)
                elif key_state == 'keyup':
                    down_keys.remove(key)
                else:
                    assert False, key_state
        x, y, left_click, right_click, key_events = action
        mapping_dict_with_key_states[(record_num, image_num)] = (x, y, left_click, right_click, list(down_keys))
    # Save dictionary using pickle
    print(f"Saving mapping with {len(mapping_dict_with_key_states)} entries...")
    with open('image_action_mapping_with_key_states.pkl', 'wb') as f:
        pickle.dump(mapping_dict_with_key_states, f)
    print("Done!")


if __name__ == "__main__":
    create_mapping()
