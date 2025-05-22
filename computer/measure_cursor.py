import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from train_coordinate_predictor import CoordinateTrainer
from latent_diffusion.ldm.util import instantiate_from_config

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test coordinate prediction model on dataset')
    parser.add_argument('--config', type=str, default="DEBUG.yaml", help='Path to the configuration file')
    parser.add_argument('--checkpoint', type=str, default="cursor_position_model.ckpt", help='Path to the model checkpoint')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to test')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--save_dir', type=str, default="test_results", help='Directory to save results')
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = CoordinateTrainer.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test the model
    #print(f"Testing on {len(indices)} samples...")
    result = model.predict('../data/data_processing/train_dataset/record_0/image_0.png', 0, 0, return_overlay=True)
    
    # Get prediction from result
    pred_x = result["predicted_x"]
    pred_y = result["predicted_y"]
    print (result)

if __name__ == "__main__":
    main() 
