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
        default=os.path.expanduser("~/train_dataset_encoded_online"),
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
        default=os.path.expanduser("~/train_dataset_encoded_online/train_dataset.target_frames.csv"),
        help="CSV file containing record_num and image_num information"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="autoencoder/saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384_mar10_keyboard_1024_768_4.5e7_cont_1e5_forui_real_4.5e6_4.5e7_gan_p1.0/model-step=720000.ckpt",
        help="Path to autoencoder model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="autoencoder/config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard_1024_768.4.5e7.cont.1e5.forui.real.4.5e7.gan.p1.0.yaml",
        help="Path to autoencoder model config file"
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=100,
        help="Number of videos to process (stops after processing this many unique videos)"
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
def encode_frame(model, frame_image):
    """
    Encode a single frame image to latent space.
    
    Args:
        model: The autoencoder model
        frame_image: PIL Image or numpy array of frame
    
    Returns:
        numpy array: Encoded latent vector
    """
    # Normalize image
    if isinstance(frame_image, np.ndarray):
        frame_image = Image.fromarray(frame_image)
    
    image_tensor = normalize_image(frame_image)
    image_tensor = torch.unsqueeze(image_tensor, dim=0)
    image_tensor = rearrange(image_tensor, 'b h w c -> b c h w').to(device)
    
    # Encode to latent
    latent = model.encode(image_tensor).sample()
    
    # Convert to numpy
    latent_np = latent.squeeze(0).cpu().numpy()
    
    return latent_np


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
    
    # Process videos in order until we've processed num_videos
    videos_to_process = unique_records[:args.num_videos]
    print(f"Processing {len(videos_to_process)} videos")
    
    # Collect all frames to encode
    all_frames = []
    video_frame_pairs = []
    processed_videos = []
    
    print("Extracting frames from videos...")
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
            
            # Process ALL frames from this video
            for frame_idx in frame_indices:
                frame_time = frame_idx / fps
                frame_rgb = video.get_frame(frame_time)
                all_frames.append(frame_rgb)
                video_frame_pairs.append((record_num, frame_idx))
            
            processed_videos.append(record_num)
            video.close()
        except Exception as e:
            print(f"Error processing video {record_num} ({video_path}): {e}")
            continue
    
    print(f"Extracted {len(all_frames)} frames total")
    
    if not all_frames:
        print("Error: No frames were extracted. Please check your video files.")
        return
    
    # Encode frames in batches
    print("Encoding frames to latent space...")
    all_latents = []
    
    for i in tqdm(range(0, len(all_frames), args.batch_size), desc="Encoding batches"):
        batch_frames = all_frames[i:i+args.batch_size]
        batch_latents = []
        
        for frame in batch_frames:
            try:
                latent = encode_frame(model, frame)
                batch_latents.append(latent)
            except Exception as e:
                print(f"Error encoding frame: {e}")
                continue
        
        all_latents.extend(batch_latents)
    
    print(f"Encoded {len(all_latents)} frames to latent space")
    
    if not all_latents:
        print("Error: No latents were encoded.")
        return
    
    # Get dimensions from first latent
    sample_latent = all_latents[0]
    num_channels = sample_latent.shape[0]
    
    print(f"Latent vectors have {num_channels} channels")
    print(f"Sample latent shape: {sample_latent.shape}")
    
    # Initialize arrays to store running sum and sum of squares
    sum_values = np.zeros(num_channels)
    sum_squares = np.zeros(num_channels)
    total_elements = 0
    
    # Compute statistics
    print("Computing statistics...")
    for latent in tqdm(all_latents, desc="Processing latents"):
        # Compute sum per channel
        channel_sum = np.sum(latent, axis=tuple(range(1, latent.ndim)))
        sum_values += channel_sum
        
        # Compute sum of squares per channel
        channel_sum_squares = np.sum(np.square(latent), axis=tuple(range(1, latent.ndim)))
        sum_squares += channel_sum_squares
        
        # Count elements per channel
        n_elements = np.prod(latent.shape[1:])
        total_elements += n_elements
    
    # Calculate mean and std
    mean = sum_values / total_elements
    variance = (sum_squares / total_elements) - np.square(mean)
    std = np.sqrt(variance)
    
    # Convert to Python lists for JSON serialization
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "num_frames_processed": len(all_latents),
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
    print(f"Processed {len(all_latents)} frames from {len(processed_videos)} videos")


if __name__ == "__main__":
    main()
