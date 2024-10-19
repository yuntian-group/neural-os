import numpy as np
from latent_diffusion.ldm.models.autoencoder import VQModel
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion
from latent_diffusion.ldm.models.diffusion.ddim import DDIMSampler
import torch, torchvision, argparse
from omegaconf import OmegaConf
from einops import rearrange
import os
from PIL import Image
from data.data_processing.video_convert import create_padding_img, create_video_from_frames
from data.data_processing.datasets import normalize_image
from computer.util import load_model_from_config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def sample_model(model: VQModel, images: list, save_path: str, create_video: bool = False):

    """
    Encodes the images and decodes them.

    Conditions on ground truth frames. Useful for evaluating accuracy given perfect input sequence.

    Parameters:
        prompts: list of string prompts representing the actions.
        image_sequences: list of processed images as arrays.
        create_video: creates a video of the output frames.
    """

    os.makedirs(save_path, exist_ok=True)

    print("\u2705 Sampling model...")

    generated_frames = []

    with torch.cuda.amp.autocast(enabled=False):
        for i,(image) in enumerate(images):

            image = torch.unsqueeze(image, dim=0)
            image = rearrange(image, 'b h w c -> b c h w').to(device)

            latent = model.encode_to_prequant(image)
            image = model.decode(latent)

            image = torch.clamp((image+1.0)/2.0, min=0.0, max=1.0)
            
            grid = 255. * rearrange(image.squeeze(0).cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(grid.astype(np.uint8))
            img.save(f'{save_path}/sample_{i}.png')

    if create_video: create_video_from_frames(generated_frames, save_path + '/sample_video.mp4', fps=15)


def parse_args():
    parser = argparse.ArgumentParser(description="Runs the inference script.")

    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="what model to sample from.")
    
    parser.add_argument("--config", type=str, default="autoencoder/config.yaml",
                        help="specifies the model config to load.")
    
    parser.add_argument("--image_paths", nargs='+', required=True,
                        help="A list of image paths.")
    
    parser.add_argument("--save_path", type=str, default='autoencoder/sample_0',
                        help="where to save the output.")


    return parser.parse_args()

if __name__ == '__main__':

    """
    Small script to encode and decode a bunch of images.
    """

    args = parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt_path)
    model = model.to(device)
    
    images = torch.stack([normalize_image(str(path)) for path in args.image_paths])

    sample_model(model, images, args.save_path)