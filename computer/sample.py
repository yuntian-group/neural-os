from typing import List
import numpy as np
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion
from latent_diffusion.ldm.models.diffusion.ddim import DDIMSampler
import torch, torchvision
from omegaconf import OmegaConf
from einops import rearrange, repeat
from torchvision.utils import make_grid
from PIL import Image
from data.data_processing.video_convert import create_video_from_frames

def sample_model(model: LatentDiffusion, prompts: List[str], image_sequences: list, save_path: str, create_video: bool):

    """
    Samples the model for each prompt and saves the resulting image.
    Paramaters:
        prompts: list of prompts from the dataset.
        image_sequences: list of images as arrays.
        create_video: creates a video of the output frames.
    """

    sampler = DDIMSampler(model)
    generated_frames = []

    with torch.no_grad():
        for i,(prompt, img_seq) in enumerate(zip(prompts, image_sequences)):

            u_dict = {'c_crossattn': "", 'c_concat': img_seq}
            uc = model.get_learned_conditioning(u_dict)
            c = model.enc_concat_seq(uc, u_dict, 'c_concat')
            # uc['c_concat'] = torch.zeros(1, 3 * 7, 64, 64, device=device)

            c_dict = {'c_crossattn': prompt, 'c_concat': img_seq}
            c = model.get_learned_conditioning(c_dict)
            c = model.enc_concat_seq(c, c_dict, 'c_concat')

            #The context frames condition is dropped with probability 0.1 to allow CFG during inference.

            samples_ddim, _ = sampler.sample(S=200,
                                            conditioning=c,
                                            batch_size=1,
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            unconditional_guidance_scale=5.0,
                                            unconditional_conditioning=uc, 
                                            eta=0)
            
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
            
            grid = 255. * rearrange(x_samples_ddim.squeeze(0).cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(grid.astype(np.uint8))
            img.save(f'{save_path}/test_decoded_256x256_{i}({prompt}).png')

            generated_frames.append(img)

        if create_video: create_video_from_frames(generated_frames, save_path + '/sample_video.mp4')