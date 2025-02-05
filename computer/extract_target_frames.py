import pandas as pd
import re
from tqdm import tqdm

def extract_numbers(path):
    """Extract record and image numbers from path"""
    match = re.search(r'record_(\d+)/image_(\d+)', path)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def extract_targets():
    # Read the filtered sequences
    print("Reading filtered sequences...")
    #df = pd.read_csv('desktop_sequences_filtered.csv')
    df = pd.read_csv('desktop_sequences_filtered_with_desktop_1.5k.csv')
    
    # Extract target frames
    target_data = []
    
    print("Extracting target frames...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing targets"):
        target_img = row['Target_image']
        record_num, img_num = extract_numbers(target_img)
        target_data.append({
            'record_num': record_num,
            'image_num': img_num
        })
    
    # Create DataFrame and save
    print("Saving results...")
    target_df = pd.DataFrame(target_data)
    target_df.to_csv('desktop_sequences_filtered_with_desktop_1.5k_target_frames.csv', index=False)
    print(f"Extracted {len(target_df)} target frames")

if __name__ == "__main__":
    extract_targets()
