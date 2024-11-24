import numpy as np
import torch
import argparse
from PIL import Image
import os
from einops import rearrange
from omegaconf import OmegaConf
from computer.util import load_model_from_config
from data.data_processing.datasets import normalize_image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def modify_model_channels(model, n_channels):
    """Modify the model to use specified number of channels in latent space"""
    # Modify quant_conv (encoder side bottleneck)
    old_quant_conv = model.quant_conv
    new_quant_conv = torch.nn.Conv2d(
        old_quant_conv.in_channels,
        2 * n_channels,  # 2*embed_dim (mean and std)
        kernel_size=1
    ).to(device)
    
    # Transfer weights properly maintaining mean/std pairs
    old_channels = old_quant_conv.weight.data.shape[0] // 2  # number of channels in original model
    # Take first n_channels from means (first half)
    new_quant_conv.weight.data[:n_channels] = old_quant_conv.weight.data[:n_channels]
    new_quant_conv.bias.data[:n_channels] = old_quant_conv.bias.data[:n_channels]
    # Take first n_channels from stds (second half)
    new_quant_conv.weight.data[n_channels:] = old_quant_conv.weight.data[old_channels:old_channels+n_channels]
    new_quant_conv.bias.data[n_channels:] = old_quant_conv.bias.data[old_channels:old_channels+n_channels]
    
    model.quant_conv = new_quant_conv

    # Modify post_quant_conv (decoder side bottleneck)
    old_post_quant_conv = model.post_quant_conv
    new_post_quant_conv = torch.nn.Conv2d(
        n_channels,  # embed_dim
        old_post_quant_conv.out_channels,
        kernel_size=1
    ).to(device)
    
    # Transfer weights for the channels we're keeping
    new_post_quant_conv.weight.data = old_post_quant_conv.weight.data[:, :n_channels]
    new_post_quant_conv.bias.data = old_post_quant_conv.bias.data.clone()
    model.post_quant_conv = new_post_quant_conv
    
    return model

@torch.no_grad()
def visualize_reconstruction(model, image_path, save_path):
    """
    Takes an input image, shows its latent representation and reconstruction.
    
    Args:
        model: The autoencoder model
        image_path: Path to input image
        save_path: Where to save visualizations
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Load and process image
    image = normalize_image(image_path)
    image = torch.unsqueeze(image, dim=0)
    image = rearrange(image, 'b h w c -> b c h w').to(device)
    
    # Get latent representation
    latent = model.encode(image).sample()
    
    # Decode back to image
    reconstruction = model.decode(latent)
    
    # Save original image
    original = torch.clamp((image+1.0)/2.0, min=0.0, max=1.0)
    original = 255. * rearrange(original.squeeze(0).cpu().numpy(), 'c h w -> h w c')
    Image.fromarray(original.astype(np.uint8)).save(os.path.join(save_path, 'original.png'))
    
    # Save latent visualization (normalize to 0-255 range)
    latent_viz = latent.squeeze(0).cpu().numpy()
    latent_viz = latent_viz - latent_viz.min()
    latent_viz = (255 * latent_viz / latent_viz.max()).astype(np.uint8)
    # Save each latent channel
    for i, channel in enumerate(latent_viz):
        Image.fromarray(channel).save(os.path.join(save_path, f'latent_channel_{i}.png'))
    
    # Save reconstruction
    reconstruction = torch.clamp((reconstruction+1.0)/2.0, min=0.0, max=1.0)
    reconstruction = 255. * rearrange(reconstruction.squeeze(0).cpu().numpy(), 'c h w -> h w c')
    Image.fromarray(reconstruction.astype(np.uint8)).save(os.path.join(save_path, 'reconstruction.png'))

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize autoencoder encoding and decoding.")
    
    parser.add_argument("--ckpt_path", type=str, default='autoencoder_kl_f16.ckpt',
                        help="Path to model checkpoint.")
    
    parser.add_argument("--config", type=str, default="config_kl4_lr4.5e6.yaml",
                        help="Path to model config.")
    
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image.")
    
    parser.add_argument("--save_path", type=str, default="visualization_kl",
                        help="Where to save visualizations.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Test different channel counts
    channel_counts = [1, 2, 3, 8, 16]
    
    for n_channels in channel_counts:
        print(f"Processing with {n_channels} channels...")
        
        # Load fresh model for each channel count
        config = OmegaConf.load(args.config)
        model = load_model_from_config(config, args.ckpt_path)
        model = model.to(device)
        model.eval()
        
        # Modify model architecture
        model = modify_model_channels(model, n_channels)
        
        # Create subdirectory for this channel count
        save_subdir = os.path.join(args.save_path, f'channels_{n_channels}')
        
        # Process image
        visualize_reconstruction(model, args.image_path, save_subdir)
    
    print(f"Visualizations saved to {args.save_path}")

if __name__ == '__main__':
    main() 
