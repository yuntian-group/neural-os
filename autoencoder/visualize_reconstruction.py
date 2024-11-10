import numpy as np
import torch
import argparse
from PIL import Image
import os
from einops import rearrange
from omegaconf import OmegaConf
from computer.util import init_and_load_model, normalize_image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    latent = model.encode_to_prequant(image)
    
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
    
    parser.add_argument("--ckpt_path", type=str, default='autoencoder.ckpt',
                        help="Path to model checkpoint.")
    
    parser.add_argument("--config", type=str, default="config_vq8.yaml",
                        help="Path to model config.")
    
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image.")
    
    parser.add_argument("--save_path", type=str, default="visualization",
                        help="Where to save visualizations.")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Load model
    config = OmegaConf.load(args.config)
    model = init_and_load_model(config, args.ckpt_path)
    model = model.to(device)
    model.eval()
    
    # Process image
    visualize_reconstruction(model, args.image_path, args.save_path)
    print(f"Visualizations saved to {args.save_path}") 
