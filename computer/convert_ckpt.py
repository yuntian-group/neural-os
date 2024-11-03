import torch
import shutil
import os
from datetime import datetime

def convert_checkpoint_for_posmap(checkpoint_path):
    # Create backup
    backup_path = checkpoint_path + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy2(checkpoint_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Get the key we need to modify
    key = "model.diffusion_model.input_blocks.0.0.weight"
    old_weight = checkpoint['state_dict'][key]
    print(f"Old weight shape: {old_weight.shape}")
    
    # Initialize new weight with same std as old weight
    new_shape = (192, 25, 3, 3)  # New shape with position map channels
    std = old_weight.std()
    new_weight = torch.randn(new_shape) * std
    
    # Copy over the old weights
    #new_weight[:, :24, :, :] = old_weight
    
    # Initialize the new channels (position map channels)
    # The new channels are 24:29
    print(f"New weight shape: {new_weight.shape}")
    
    # Replace in checkpoint
    checkpoint['state_dict'][key] = new_weight
    
    # Save modified checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved modified checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    checkpoint_path = "oct29_fixcursor_test_15_no_deltas_1000_paths/model_test_15_no_deltas_1000_paths.ckpt"  # Replace with your checkpoint path
    convert_checkpoint_for_posmap(checkpoint_path)
