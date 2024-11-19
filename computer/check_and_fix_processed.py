import os
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
import pandas as pd
import ast
from computer.util import load_model_from_config
from latent_diffusion.ldm.util import instantiate_from_config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def normalize_image(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = (np.array(image) / 127.5 - 1.0).astype(np.float32)
    return torch.tensor(image)

def check_and_fix_processed_files(csv_path, model):
    """Check all processed files referenced in csv and fix any that are corrupted or missing"""
    print(f"Checking files referenced in {csv_path}")
    
    # Read CSV
    data = pd.read_csv(csv_path)
    image_seq_paths = data["Image_seq_cond_path"].apply(ast.literal_eval).to_list()
    targets = data['Target_image'].to_list()
    
    # Collect all unique image paths
    all_paths = set()
    for seq in image_seq_paths:
        all_paths.update(seq)
    all_paths.update(targets)
    
    # Add padding.png if it exists
    padding_path = "train_dataset/padding.png"
    if os.path.exists(padding_path):
        all_paths.add(padding_path)
    
    print(f"Found {len(all_paths)} unique images to check")
    
    # Check each file
    problems_found = 0
    for img_path in tqdm(all_paths):
        processed_path = img_path.replace('train_dataset/', 'train_dataset_encoded/').replace('.png', '.npy')
        
        needs_processing = False
        try:
            # Try to load the processed file
            if not os.path.exists(processed_path):
                print(f"\nMissing: {processed_path}")
                needs_processing = True
            else:
                try:
                    np.load(processed_path)
                except Exception as e:
                    print(f"\nCorrupted: {processed_path}")
                    print(f"Error: {e}")
                    needs_processing = True
        except Exception as e:
            print(f"\nError checking {processed_path}: {e}")
            needs_processing = True
        
        if needs_processing:
            problems_found += 1
            try:
                # Process the image
                image = normalize_image(img_path)
                image = torch.unsqueeze(image, dim=0)
                image = rearrange(image, 'b h w c -> b c h w').to(device)
                
                # Get latent representation
                with torch.no_grad():
                    posterior = model.encode(image)
                    latent = posterior.sample()
                    
                    # Special handling for padding.png
                    if os.path.basename(img_path) == 'padding.png':
                        latent = torch.zeros_like(latent)
                    latent = latent.squeeze(0)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(processed_path), exist_ok=True)
                
                # Save the processed latent
                np.save(processed_path, latent.cpu().numpy())
                print(f"Fixed: {processed_path}")
            except Exception as e:
                print(f"Failed to fix {processed_path}: {e}")
    
    print(f"\nCheck complete. Found and attempted to fix {problems_found} problems.")

if __name__ == "__main__":
    # Load model
    print("Loading model...")
    config = OmegaConf.load("autoencoder_config_kl4_lr4.5e6_load_acc1.yaml")
    model = load_model_from_config(config, "autoencoder_saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_model-603000.ckpt")
    model = model.to(device)
    model.eval()
    
    # Check files for each dataset
    csv_files = [
        "train_dataset/train_dataset_14frames_firstframe_allframes.csv",
        # Add other CSV files as needed
    ]
    
    for csv_path in csv_files:
        check_and_fix_processed_files(csv_path, model) 