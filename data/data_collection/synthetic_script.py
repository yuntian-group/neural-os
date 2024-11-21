from record_script import record
import time
import pyautogui
from synthetic_mouse_path import generate_multiple_trajectories
import random
import numpy as np
import gc
import os
import shutil

random.seed(1234)
np.random.seed(1234)
base_directory = '/app/raw_data'
def create_synthetic_dataset(n=1, batch_size=100, duration=12):
    screen_width = int(os.getenv('SCREEN_WIDTH', 1024))
    screen_height = int(os.getenv('SCREEN_HEIGHT', 768))
    
    # Create batches to manage memory better
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        print(f"Processing batch {batch_start} to {batch_end}")
        
        # Generate trajectories for this batch only
        trajectories = generate_multiple_trajectories(batch_end - batch_start, 
                                                   screen_width, screen_height)
        
        # Create a new directory for this batch
        batch_dir = f'raw_data_batch_{batch_start}'
        #os.makedirs(batch_dir, exist_ok=True)
        
        for i, trajectory in enumerate(trajectories):
            record_idx = batch_start + i
            record(batch_dir, f'record_{record_idx}', duration=duration, 
                   trajectory=trajectory)
            
            # Force garbage collection after each recording
            gc.collect()
            
            # Small delay to let resources clean up
            time.sleep(0.1)
        
        # Move files to final location and clean up batch directory
        for root, dirs, files in os.walk(os.path.join(base_directory, batch_dir)):
            for file in files:
                src = os.path.join(root, file)
                rel_path = os.path.relpath(root, os.path.join(base_directory, batch_dir))
                dst = os.path.join(base_directory, rel_path, file)
                
                # Ensure destination directory exists
                dst_dir = os.path.dirname(dst)
                os.makedirs(dst_dir, exist_ok=True)
                
                try:
                    print(f"Moving {file} from {src} to {dst}")
                    shutil.move(src, dst)
                except Exception as e:
                    print(f"Error moving {file}: {str(e)}")
        
        shutil.rmtree(os.path.join(base_directory, batch_dir))
        
        # Additional cleanup
        gc.collect()
        time.sleep(1.0)  # Longer pause between batches

if __name__ == "__main__":
    # Ensure raw_data directory exists
    os.makedirs('raw_data', exist_ok=True)
    
    # Run with batch size of 100
    create_synthetic_dataset(2, batch_size=1)
