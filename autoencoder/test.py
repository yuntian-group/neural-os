import torch
import numpy as np
import os
from omegaconf import OmegaConf
from computer.util import load_model_from_config
from PIL import Image
from einops import rearrange


device = torch.device('cpu')
config = OmegaConf.load("config_kl4_lr4.5e6_load_acc1.yaml")
model = load_model_from_config(config, "saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384/model-354000.ckpt").to(device)
debug_dir = 'debug_autoencoder'
os.makedirs(debug_dir, exist_ok=True)
output_folder = '../computer/train_dataset_encoded/record_10'
input_folder = '../data/data_processing/train_dataset/record_10'
batch_files = [f for f in sorted(os.listdir(input_folder)) if f.endswith('.png')]
batch_files = batch_files[:3]
# Load saved latents and decode
loaded_latents = []
for img_file in batch_files:
    latent_path = os.path.join(output_folder, img_file.replace('.png', '.npy'))
    loaded_latent = np.load(latent_path)
    loaded_latents.append(torch.from_numpy(loaded_latent))

# Stack and move to device
loaded_latents = torch.stack(loaded_latents).to(device)

# Decode loaded latents back to images
with torch.no_grad():
    reconstructions = model.decode(loaded_latents)


images = []
for img_file in batch_files:
    image = Image.open(os.path.join(input_folder, img_file))
    if not image.mode == "RGB":
        image = image.convert("RGB")
    images.append(np.array(image))

# Stack and process all images at once on CPU
images = np.stack(images)
images = (images / 127.5 - 1.0).astype(np.float32)
images = torch.tensor(images)
images = rearrange(images, 'b h w c -> b c h w')
# Save original and reconstructed images side by side
for idx, (orig, recon, fname) in enumerate(zip(images, reconstructions, batch_files)):
    # Convert to numpy and move to CPU
    orig = orig.cpu().numpy()
    recon = recon.cpu().numpy()
    print (orig.shape, recon.shape)
    
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
    Image.fromarray(comparison).save( os.path.join(debug_dir, f'debug_{idx}_{fname}')
                )
        
