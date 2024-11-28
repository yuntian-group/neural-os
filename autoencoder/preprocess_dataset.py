import numpy as np
import torch
import argparse
from PIL import Image
import os
from einops import rearrange
from omegaconf import OmegaConf
from computer.util import load_model_from_config
from data.data_processing.datasets import normalize_image
from tqdm import tqdm
import shutil

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def process_folder(model, input_folder, output_folder, batch_size=16, debug_first_batch=False):
    """Process all images in a folder through the encoder in batches"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Collect all PNG files
    img_files = [f for f in sorted(os.listdir(input_folder)) if f.endswith('.png')]
    
    # Process in batches
    for i in tqdm(range(0, len(img_files), batch_size)):
        batch_files = img_files[i:i+batch_size]
        images = []
        
        for img_file in batch_files:
            # Load and process image
            image_path = os.path.join(input_folder, img_file)
            image = normalize_image(image_path)
            image = torch.unsqueeze(image, dim=0)
            image = rearrange(image, 'b h w c -> b c h w').to(device)
            images.append(image)
        
        # Stack images into a batch
        images = torch.cat(images, dim=0)
        
        # Get latent representations through full encoding process
        posterior = model.encode(images)
        latents = posterior.sample()  # Sample from the posterior
        
        # Save each latent as numpy array
        for img_file, latent in zip(batch_files, latents):
            latent_path = os.path.join(output_folder, img_file.replace('.png', '.npy'))
            np.save(latent_path, latent.cpu().numpy())
            
        # Debug first batch after saving
        if debug_first_batch and i == 0:
            debug_dir = os.path.join(output_folder, 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            
            # Load saved latents and decode
            loaded_latents = []
            for img_file in batch_files:
                latent_path = os.path.join(output_folder, img_file.replace('.png', '.npy'))
                loaded_latent = np.load(latent_path)
                loaded_latents.append(torch.from_numpy(loaded_latent))
            
            # Stack and move to device
            loaded_latents = torch.stack(loaded_latents).to(device)
            
            # Decode loaded latents back to images
            reconstructions = model.decode(loaded_latents)
            
            # Save original and reconstructed images side by side
            for idx, (orig, recon, fname) in enumerate(zip(images, reconstructions, batch_files)):
                # Convert to numpy and move to CPU
                orig = orig.cpu().numpy()
                recon = recon.cpu().numpy()
                
                # Denormalize from [-1,1] to [0,255]
                orig = (orig + 1.0) * 127.5
                recon = (recon + 1.0) * 127.5
                
                # Clip values to valid range
                orig = np.clip(orig, 0, 255).astype(np.uint8)
                recon = np.clip(recon, 0, 255).astype(np.uint8)
                
                # Rearrange from BCHW to HWC
                orig = np.transpose(orig, (1,2,0))
                recon = np.transpose(recon, (1,2,0))
                
                # Create side-by-side comparison
                comparison = np.concatenate([orig, recon], axis=1)
                
                # Save comparison image
                Image.fromarray(comparison).save(
                    os.path.join(debug_dir, f'debug_{idx}_{fname}')
                )
            print(f"\nDebug visualizations saved to {debug_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-process dataset using trained encoder.")
    
    parser.add_argument("--ckpt_path", type=str, 
                        default="saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384/model-354000.ckpt",
                        help="Path to model checkpoint.")
                        #default="saved_kl4_bsz8_acc8_lr4.5e6_load_acc1/model-603000.ckpt",
                        #help="Path to model checkpoint.")
    
    parser.add_argument("--config", type=str, 
                        default="config_kl4_lr4.5e6_load_acc1.yaml",
                        help="Path to model config.")
    
    parser.add_argument("--input_dir", type=str, 
                        default="train_dataset",
                        help="Path to input dataset directory.")
    
    parser.add_argument("--output_dir", type=str, 
                        default="train_dataset_encoded",
                        help="Where to save processed dataset.")
    
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing images.")
    
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index for processing folders (inclusive)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index for processing folders (exclusive)")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Load model
    print("Loading model...")
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt_path)
    model = model.to(device)
    model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process root padding.png only in the first job
    if args.start_idx is None or args.start_idx == 0:
        root_padding = os.path.join(args.input_dir, 'padding.png')
        if os.path.exists(root_padding):
            print("Processing root padding.png...")
            # Load and process padding image to get correct shape
            image = normalize_image(root_padding)
            image = torch.unsqueeze(image, dim=0)
            image = rearrange(image, 'b h w c -> b c h w').to(device)
            
            # Get latent shape through encoder
            posterior = model.encode(image)
            latent = posterior.sample()
            
            # Set all values to 0 and save
            latent = torch.zeros_like(latent).squeeze(0)
            np.save(os.path.join(args.output_dir, 'padding.npy'), latent.cpu().numpy())
    
    # Get sorted list of record folders
    record_folders = sorted([f for f in os.listdir(args.input_dir) if f.startswith('record_')])
    
    # Apply folder range if specified
    if args.start_idx is not None and args.end_idx is not None:
        record_folders = record_folders[args.start_idx:args.end_idx]
    
    print(f"Processing dataset (folders {args.start_idx} to {args.end_idx})...")
    for folder in tqdm(record_folders):
        input_folder = os.path.join(args.input_dir, folder)
        output_folder = os.path.join(args.output_dir, folder)
        
        if os.path.isdir(input_folder):
            print(f"\nProcessing {folder}...")
            process_folder(model, input_folder, output_folder, args.batch_size)
            
            # Copy any non-image files (like CSV) directly
            for file in os.listdir(input_folder):
                if not file.endswith('.png'):
                    src = os.path.join(input_folder, file)
                    dst = os.path.join(output_folder, file)
                    shutil.copy2(src, dst)
    
    print(f"\nProcessed dataset saved to {args.output_dir}") 
