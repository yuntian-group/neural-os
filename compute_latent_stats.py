import numpy as np
import os
import argparse
from tqdm import tqdm
import json
from glob import glob
import pandas as pd
import random
import torch
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from moviepy.editor import VideoFileClip
from computer.util import load_model_from_config
from data.data_processing.datasets import normalize_image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute mean and std of latent vectors by encoding raw videos")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=os.path.expanduser("~/neuralos-demo-newdata-autoencoder/train_dataset_encoded_online"),
        help="Directory containing raw .mp4 video files"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="latent_stats.json",
        help="Output file to save mean and std values"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=os.path.expanduser("~/neuralos-demo-newdata-autoencoder/train_dataset_encoded_online/train_dataset.target_frames.csv"),
        help="CSV file containing record_num and image_num information"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="autoencoder/saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384_mar10_keyboard_1024_768_4.5e7_cont_1e5_forui_real_4.5e6_4.5e7_gan_p1.0_newdata_finalrun_balanced/model-step=004000.ckpt",
        help="Path to autoencoder model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="autoencoder/config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard_1024_768.4.5e7.cont.1e5.forui.real.4.5e7.gan.p1.0.newdata.finalrun.balanced.yaml",
        help="Path to autoencoder model config file"
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=-1,
        help="Number of videos to process (stops after processing this many unique videos). Use -1 to process all videos."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding frames"
    )
    return parser.parse_args()


@torch.no_grad()
def encode_batch(model, frames):
    """
    Encode a batch of frames to latent space.
    
    Args:
        model: The autoencoder model
        frames: List of PIL Images or numpy arrays
    
    Returns:
        list of numpy arrays: Encoded latent vectors
    """
    # Normalize and stack frames into a batch
    batch_tensors = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        image_tensor = normalize_image(frame)
        batch_tensors.append(image_tensor)
    
    # Stack into batch: [B, H, W, C]
    batch_tensor = torch.stack(batch_tensors, dim=0)
    # Rearrange to [B, C, H, W]
    batch_tensor = rearrange(batch_tensor, 'b h w c -> b c h w').to(device)
    
    # Encode batch to latent
    latents = model.encode(batch_tensor).sample()
    
    # Convert to numpy and return as list
    latents_np = [latent.cpu().numpy() for latent in latents]
    
    return latents_np


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Loading model from checkpoint: {args.ckpt_path}")
    print(f"Using config: {args.config}")
    
    # Load model
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt_path)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Read CSV file to get record_num and image_num pairs
    print(f"Reading CSV file: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    df['record_num'] = df['record_num'].astype(int)
    df['image_num'] = df['image_num'].astype(int)
    
    # Group by record_num to get unique videos and their frames
    video_frames_dict = {}
    for _, row in df.iterrows():
        record_num = row['record_num']
        image_num = row['image_num']
        if record_num not in video_frames_dict:
            video_frames_dict[record_num] = []
        video_frames_dict[record_num].append(image_num)
    
    # Get unique record numbers in order they appear in CSV
    unique_records = df['record_num'].unique()
    print(f"Found {len(unique_records)} unique videos in CSV")
    
    # Process videos in order until we've processed num_videos (or all if num_videos == -1)
    if args.num_videos == -1:
        videos_to_process = unique_records
        print(f"Processing all {len(videos_to_process)} videos")
    else:
        videos_to_process = unique_records[:args.num_videos]
        print(f"Processing {len(videos_to_process)} videos")
    
    # Initialize statistics accumulators (will be initialized after first latent)
    sum_values = None
    sum_squares = None
    total_elements = 0
    num_frames_processed = 0
    processed_videos = []
    batch_buffer = []
    
    print("Extracting and encoding frames from videos...")
    for record_num in tqdm(videos_to_process, desc="Processing videos"):
        # Get video path
        video_path = os.path.join(args.data_dir, f"{record_num}.mp4")
        
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}, skipping")
            continue
        
        try:
            video = VideoFileClip(video_path)
            fps = video.fps
            duration = video.duration
            num_frames_total = int(fps * duration)
            
            # Get all frames for this video from CSV
            frame_indices = video_frames_dict[record_num]
            # Remove duplicates and sort
            frame_indices = sorted(set(frame_indices))
            # Filter out invalid frame indices
            frame_indices = [idx for idx in frame_indices if 0 <= idx < num_frames_total]
            
            if not frame_indices:
                print(f"Warning: No valid frames found for video {record_num}, skipping")
                video.close()
                continue
            
            # Process ALL frames from this video - batch encode for efficiency
            for frame_idx in frame_indices:
                frame_time = frame_idx / fps
                frame_rgb = video.get_frame(frame_time)
                batch_buffer.append(frame_rgb)
                
                # When batch buffer is full, encode batch and accumulate statistics
                if len(batch_buffer) >= args.batch_size:
                    try:
                        latents = encode_batch(model, batch_buffer)
                        
                        # Initialize accumulators on first batch
                        if sum_values is None:
                            num_channels = latents[0].shape[0]
                            sum_values = np.zeros(num_channels)
                            sum_squares = np.zeros(num_channels)
                            print(f"Latent vectors have {num_channels} channels")
                            print(f"Sample latent shape: {latents[0].shape}")
                        
                        # Accumulate statistics for each latent in batch
                        for latent in latents:
                            channel_sum = np.sum(latent, axis=tuple(range(1, latent.ndim)))
                            sum_values += channel_sum
                            
                            channel_sum_squares = np.sum(np.square(latent), axis=tuple(range(1, latent.ndim)))
                            sum_squares += channel_sum_squares
                            
                            n_elements = np.prod(latent.shape[1:])
                            total_elements += n_elements
                            num_frames_processed += 1
                    except Exception as e:
                        print(f"Error encoding batch: {e}")
                    
                    batch_buffer = []
            
            processed_videos.append(record_num)
            video.close()
        except Exception as e:
            print(f"Error processing video {record_num} ({video_path}): {e}")
            continue
    
    # Process remaining frames in buffer
    if batch_buffer:
        try:
            latents = encode_batch(model, batch_buffer)
            
            # Initialize accumulators on first batch (if buffer had first frames)
            if sum_values is None:
                num_channels = latents[0].shape[0]
                sum_values = np.zeros(num_channels)
                sum_squares = np.zeros(num_channels)
                print(f"Latent vectors have {num_channels} channels")
                print(f"Sample latent shape: {latents[0].shape}")
            
            # Accumulate statistics for each latent in batch
            for latent in latents:
                channel_sum = np.sum(latent, axis=tuple(range(1, latent.ndim)))
                sum_values += channel_sum
                
                channel_sum_squares = np.sum(np.square(latent), axis=tuple(range(1, latent.ndim)))
                sum_squares += channel_sum_squares
                
                n_elements = np.prod(latent.shape[1:])
                total_elements += n_elements
                num_frames_processed += 1
        except Exception as e:
            print(f"Error encoding final batch: {e}")
    
    if num_frames_processed == 0:
        print("Error: No frames were encoded. Please check your video files.")
        return
    
    print(f"Encoded {num_frames_processed} frames to latent space")
    
    # Calculate mean and std
    mean = sum_values / total_elements
    variance = (sum_squares / total_elements) - np.square(mean)
    std = np.sqrt(variance)
    
    # Convert to Python lists for JSON serialization
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "num_frames_processed": num_frames_processed,
        "num_files_processed": len(processed_videos),
        "total_elements_per_channel": int(total_elements),
        "sampled_records": sorted([int(v) for v in processed_videos])  # Convert numpy int64 to Python int
    }
    
    # Save stats to file
    with open(args.output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to {args.output_file}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    print(f"Processed {num_frames_processed} frames from {len(processed_videos)} videos")


if __name__ == "__main__":
    main()
