from record_script import record
import asyncio
import time
import pyautogui
from synthetic_mouse_path import generate_multiple_trajectories, move_mouse_through_trajectory
import random
import numpy as np
import gc
import os
import shutil

random.seed(1234)
np.random.seed(1234)

async def create_synthetic_dataset(n=1, batch_size=100):
    screen_width, screen_height = pyautogui.size()
    
    # Create batches to manage memory better
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        print(f"Processing batch {batch_start} to {batch_end}")
        
        # Generate trajectories for this batch only
        trajectories = generate_multiple_trajectories(batch_end - batch_start, 
                                                   screen_width, screen_height)
        
        # Create a new directory for this batch
        batch_dir = f'raw_data_batch_{batch_start}'
        os.makedirs(batch_dir, exist_ok=True)
        
        for i, trajectory in enumerate(trajectories):
            record_idx = batch_start + i
            await record(batch_dir, f'record_{record_idx}', duration=12, 
                        function_to_record=move_mouse_through_trajectory, 
                        fn_args=(trajectory,))
            
            # Force garbage collection after each recording
            gc.collect()
            
            # Small delay to let resources clean up
            await asyncio.sleep(0.1)
        
        # Move files to final location and clean up batch directory
        print(f"\nMoving files from {batch_dir} to raw_data/")
        
        # First, list all files to move
        for root, dirs, files in os.walk(batch_dir):
            print(f"\nExploring directory: {root}")
            print(f"Found directories: {dirs}")
            print(f"Found files: {files}")
            
            for file in files:
                src = os.path.join(root, file)
                rel_path = os.path.relpath(root, batch_dir)
                dst = os.path.join('raw_data', rel_path, file)
                
                print(f"\nMoving file:")
                print(f"From: {src}")
                print(f"To: {dst}")
                
                # Ensure destination directory exists
                dst_dir = os.path.dirname(dst)
                os.makedirs(dst_dir, exist_ok=True)
                
                try:
                    shutil.move(src, dst)
                    print(f"Successfully moved {file}")
                except Exception as e:
                    print(f"Error moving {file}: {str(e)}")
        
        print(f"\nCleaning up {batch_dir}")
        shutil.rmtree(batch_dir)
        
        # Additional cleanup
        gc.collect()
        await asyncio.sleep(1.0)  # Longer pause between batches

if __name__ == "__main__":
    # Ensure raw_data directory exists
    os.makedirs('raw_data', exist_ok=True)
    
    # Run with batch size of 100
    asyncio.run(create_synthetic_dataset(4000, batch_size=100))
