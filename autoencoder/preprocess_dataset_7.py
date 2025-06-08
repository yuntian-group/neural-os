import numpy as np
import torch
import argparse
from PIL import Image
import os
import io
from einops import rearrange
from omegaconf import OmegaConf
from computer.util import load_model_from_config
from data.data_processing.datasets import normalize_image
from tqdm import tqdm
import shutil
import multiprocessing as mp
from functools import partial
import webdataset as wds
from moviepy.editor import VideoFileClip


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def process_record(model, video_file, video_dir, output_dir, batch_size=16, debug_first_batch=False):
    """Process a single record's tar file through the encoder in batches"""
    #input_tar_path = os.path.join(input_dir, record_file)
    input_video_path = os.path.join(video_dir, video_file)
    output_tar_path = os.path.join(output_dir, video_file.split('.')[0] + '.tar')
    
    # Make sure we have an input tar file
    if not os.path.exists(input_video_path):
        print(f"Skipping {input_video_path} - file not found")
        return
    
    # Create output tar writer
    sink = wds.TarWriter(output_tar_path)
    
    # Load dataset from tar file
    #dataset = wds.WebDataset(input_tar_path).decode()
    
    # Process in batches
    batch_images = []
    batch_keys = []
    
    # Flag to enable debug on the first batch if requested
    first_batch = True
    
    # Load video
    with VideoFileClip(input_video_path) as video:
        fps = video.fps
        duration = video.duration
        # Make FPS check a warning instead of hard assertion
        if fps != 15:
            print(f"Warning: Expected FPS of 15, got {fps}")
            assert False
        for image_num in tqdm(range(0, int(fps * duration)), desc=f"Processing {video_file}"):
            frame = video.get_frame(image_num / fps)        
            # Get key and image data
            key = str(image_num)
            image_data = frame  # Already a numpy array
            
            # Add to batch
            batch_images.append(image_data)
            batch_keys.append(key)
            
            # Process batch when it reaches the specified size
            if len(batch_images) >= batch_size:
                process_batch(model, batch_images, batch_keys, sink, 
                            debug=debug_first_batch and first_batch, 
                            video_file=video_file, 
                            output_dir=output_dir)
                
                # Reset batch
                batch_images = []
                batch_keys = []
                first_batch = False
        
        # Process any remaining images
        if batch_images:
            process_batch(model, batch_images, batch_keys, sink, 
                        debug=debug_first_batch and first_batch, 
                        video_file=video_file, 
                        output_dir=output_dir)
    
    # Close tar writer
    sink.close()

@torch.no_grad()
def process_batch(model, images, keys, sink, debug=False, video_file=None, output_dir=None):
    """Process a batch of images through the encoder"""
    # Stack and process all images
    image_array = np.stack(images)
    
    # Normalize to [-1, 1]
    image_array = (image_array / 127.5 - 1.0).astype(np.float32)
    
    # Convert to torch tensor
    images_tensor = torch.tensor(image_array)
    images_tensor = rearrange(images_tensor, 'b h w c -> b c h w')
    
    # Move to device for inference
    images_tensor = images_tensor.to(device)
    
    # Encode images
    posterior = model.encode(images_tensor)
    latents = posterior.sample()  # Sample from the posterior
    
    # Move back to CPU for saving
    latents = latents.cpu()
    
    # Save each latent to the tar file
    for key, latent in zip(keys, latents):
        # Convert latent to bytes
        latent_bytes = io.BytesIO()
        np.save(latent_bytes, latent.numpy())
        latent_bytes.seek(0)
        
        # Write to tar
        sample = {
            "__key__": key,
            "npy": latent_bytes.getvalue(),
        }
        sink.write(sample)
    
    # Debug first batch if requested
    if debug:
        debug_dir = os.path.join(output_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        # Decode latents back to images
        reconstructions = model.decode(latents.to(device))
        
        # Save original and reconstructed images side by side
        for idx, (orig, recon) in enumerate(zip(images_tensor, reconstructions)):
            # Convert to numpy and move to CPU
            orig = orig.cpu().numpy()
            recon = recon.cpu().numpy()
            
            # Denormalize from [-1,1] to [0,255]
            orig = (orig + 1.0) * 127.5
            recon = (recon + 1.0) * 127.5
            
            # Clip values to valid range
            orig = np.clip(orig, 0, 255).astype(np.uint8)
            recon = np.clip(recon, 0, 255).astype(np.uint8)
            
            # Rearrange from CHW to HWC
            orig = np.transpose(orig, (1,2,0))
            recon = np.transpose(recon, (1,2,0))
            
            # Create side-by-side comparison
            comparison = np.concatenate([orig, recon], axis=1)
            
            # Save comparison image
            Image.fromarray(comparison).save(
                os.path.join(debug_dir, f'debug_{video_file}_{idx}_{keys[idx]}.png')
            )
        print(f"\nDebug visualizations saved to {debug_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-process dataset using trained encoder.")
    
    parser.add_argument("--ckpt_path", type=str, 
                        default="saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384_mar10_keyboard_init_16_cont_mar15_acc1_cont_1e6_cont_2e7_cont/model-2076000.ckpt",
                        help="Path to model checkpoint.")
    
    parser.add_argument("--config", type=str, 
                        default="config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard_init_16_contmar15_acc1.yaml",
                        help="Path to model config.")
    
    parser.add_argument("--input_dir", type=str, 
                        default="../data/data_processing/train_dataset_may20_7_webdataset",
                        help="Path to input dataset directory with WebDataset tar files.")
    parser.add_argument("--video_dir", type=str, default="/home/yuntian/scratch/raw_data_may20_7/raw_data/videos",
                        help="Path to raw video files. If provided, will process videos directly instead of tar files.")
    
    parser.add_argument("--output_dir", type=str, 
                        default="./train_dataset_may20_7_webdataset_encoded",
                        help="Where to save processed dataset.")
    
    parser.add_argument("--batch_size", type=int, default=150,
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
        root_padding = os.path.join(args.input_dir, 'padding.npy')
        if os.path.exists(root_padding):
            print("Processing root padding.npy...")
            # Load padding data
            padding_data = np.load(root_padding, allow_pickle=True)
            
            # Normalize and convert to tensor
            padding_image = (padding_data / 127.5 - 1.0).astype(np.float32)
            padding_tensor = torch.tensor(padding_image).unsqueeze(0)
            padding_tensor = rearrange(padding_tensor, 'b h w c -> b c h w').to(device)
            
            # Get latent shape through encoder
            posterior = model.encode(padding_tensor)
            latent = posterior.sample()
            
            # Set all values to 0 and save
            latent = torch.zeros_like(latent).squeeze(0)
            np.save(os.path.join(args.output_dir, 'padding.npy'), latent.cpu().numpy())
    
    # Get sorted list of record folders
    video_files = sorted([f for f in os.listdir(args.video_dir) if f.startswith('record_')])
    #record_files = sorted([f for f in os.listdir(args.input_dir) if f.startswith('record_')])
    
    # Apply folder range if specified
    if args.start_idx is not None and args.end_idx is not None:
        #record_files = record_files[args.start_idx:args.end_idx]
        video_files = video_files[args.start_idx:args.end_idx]
    print(f"Processing dataset (records {args.start_idx} to {args.end_idx})...")
    
    # Process each record in sequence
    #for record_file in tqdm(record_files, desc="Processing records"):
    for video_file in tqdm(video_files, desc="Processing records"):
        process_record(
            model, 
            video_file, 
            args.video_dir, 
            args.output_dir, 
            args.batch_size,
            #debug_first_batch=(video_file == video_files[0])
            debug_first_batch=False
        )
    
    # Copy the metadata files (CSV and PKL)
    # Copy any non-image files (like CSV) directly
    for file in os.listdir(args.input_dir):
        if (not file.endswith('.tar')) and (not file.endswith('.npy')):
            src = os.path.join(args.input_dir, file)
            dst = os.path.join(args.output_dir, file)
            shutil.copy2(src, dst)
    
    print(f"\nProcessed dataset saved to {args.output_dir}")
