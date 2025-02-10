#!/usr/bin/env python3

import os
import re
import shutil

def find_latest_checkpoint(folder):
    """
    Return the name of the checkpoint file in `folder` with the highest step number.
    If no checkpoint is found, return None.
    """
    pattern = re.compile(r'model-step=(\d+)\.ckpt$')
    latest_step = -1
    latest_file = None

    # List everything in the folder
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_file = filename

    return latest_file

def main():
    # You can change the values [4, 8, 16, 32] to whatever set of context sizes you want to handle.
    context_sizes = [4, 8, 16, 32]
    
    for size in context_sizes:
        source_folder = f"saved_standard_challenging_context{size}_cont"
        dest_folder   = f"saved_standard_challenging_context{size}"
        
        if not os.path.isdir(source_folder):
            print(f"Folder '{source_folder}' does not exist. Skipping.")
            continue
        
        latest_ckpt = find_latest_checkpoint(source_folder)
        if not latest_ckpt:
            print(f"No checkpoints found in '{source_folder}'. Skipping.")
            continue
        
        # Ensure the destination folder exists
        os.makedirs(dest_folder, exist_ok=True)
        
        src_path = os.path.join(source_folder, latest_ckpt)
        dst_path = os.path.join(dest_folder, "model-step=500000.ckpt")
        
        print(f"Copying from '{src_path}' to '{dst_path}'")
        shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    main()

