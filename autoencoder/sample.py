import numpy as np
from latent_diffusion.ldm.models.autoencoder import VQModel
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion
from latent_diffusion.ldm.models.diffusion.ddim import DDIMSampler
import torch, torchvision, argparse
from omegaconf import OmegaConf
from einops import rearrange
import os
from PIL import Image
from data.processing.video_convert import create_video_from_frames
from data.processing.datasets import normalize_image, ActionsData
from computer.util import init_and_load_model, create_loss_plot, get_mse_image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def sample_model(model: VQModel, images: list, save_path: str, create_video: bool = False, targets: list = None):

    """
    Encodes the images and decodes them.

    Parameters:
        prompts: list of string prompts representing the actions.
        image_sequences: list of processed images as arrays.
        create_video: creates a video of the output frames.
        targets: list of target frames. calculates mse between the generated frames and their ground truth targets (if provided).
    """

    os.makedirs(save_path, exist_ok=True)

    print("\u2705 Sampling model...")

    generated_frames = []
    mse_losses = []

    with torch.cuda.amp.autocast(enabled=False):
        for i,(image) in enumerate(images):

            image = torch.unsqueeze(image, dim=0)
            image = rearrange(image, 'b h w c -> b c h w').to(device)

            latent = model.encode_to_prequant(image)
            image = model.decode(latent)
            if targets: mse_losses.append(get_mse_image(rearrange(image.squeeze(0).cpu(),'c h w -> h w c'), targets[i]))

            image = torch.clamp((image+1.0)/2.0, min=0.0, max=1.0)
            
            grid = 255. * rearrange(image.squeeze(0).cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(grid.astype(np.uint8))
            img.save(os.path.join(save_path, f'sample_{i}.png'))

            generated_frames.append(img)

    if create_video: create_video_from_frames(generated_frames, os.path.join(save_path, 'sample_video.mp4'), fps=15)
    if targets: create_loss_plot(mse_losses, save_path, title='Sample MSE', dot_plot=True)

@torch.no_grad()
def sample_model_from_dataset(model: VQModel, dataset: ActionsData, idxs: list, save_path: str, create_video: bool = False, targets: bool = False):

    """
    Encodes the targets in the dataset from the given indices.

    Parameters:
        dataset: dataset with __getitem__ method.
        idxs: items to sample.
        create_video: creates a video of the output frames.
        targets: computes mse plot if true.
    """

    os.makedirs(save_path, exist_ok=True)

    print("\u2705 Sampling model...")

    generated_frames = []
    mse_losses = []

    with torch.cuda.amp.autocast(enabled=False):
        for idx in idxs:

            item = dataset.__getitem__(idx)

            image = item['image']
            target = image

            image = torch.unsqueeze(image, dim=0)
            image = rearrange(image, 'b h w c -> b c h w').to(device)

            latent = model.encode_to_prequant(image)
            image = model.decode(latent)
            if targets: mse_losses.append(get_mse_image(rearrange(image.squeeze(0).cpu(),'c h w -> h w c'), target))

            image = torch.clamp((image+1.0)/2.0, min=0.0, max=1.0)
            
            grid = 255. * rearrange(image.squeeze(0).cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(grid.astype(np.uint8))
            img.save(os.path.join(save_path, f'sample_{idx}.png'))

            generated_frames.append(img)

    if create_video: create_video_from_frames(generated_frames, os.path.join(save_path, 'sample_video.mp4'), fps=15)
    if targets: create_loss_plot(mse_losses, save_path, title='Sample MSE', dot_plot=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Runs the inference script.")

    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="what model to sample from.")
    
    parser.add_argument("--config", type=str, default="config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard.yaml"),
                        help="specifies the model config to load.")
    
    parser.add_argument("--image_paths", nargs='+', required=True,
                        help="A list of image paths.")
    
    parser.add_argument("--save_path", type=str, default='sample_keyboard_mar10',
                        help="where to save the output.")


    return parser.parse_args()

if __name__ == '__main__':

    """
    Small script to encode and decode a bunch of images.
    """

    args = parse_args()

    config = OmegaConf.load(args.config)
    model = init_and_load_model(config, args.ckpt_path)
    model = model.to(device)
    
    images = torch.stack([normalize_image(str(path)) for path in args.image_paths])

    sample_model(model, images, args.save_path)