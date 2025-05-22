import torch
from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion, DDIMSampler
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import json
import os
import time
DEBUG = False

def load_model_from_config(config_path, model_name, device='cuda', load=True):
    # Load the config file
    config = OmegaConf.load(config_path)
    
    # Instantiate the model
    model = instantiate_from_config(config.model)
    
    # Download the model file from Hugging Face
    if load:
        model_file = hf_hub_download(repo_id=model_name, filename="model.safetensors", token=os.getenv('HF_TOKEN'))
        print(f"Loading model from {model_name}")
        # Load the state dict
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval()
    return model

def sample_frame(model: LatentDiffusion, prompt: str, image_sequence: torch.Tensor, pos_maps=None, leftclick_maps=None):
    sampler = DDIMSampler(model)
    
    with torch.no_grad():
        #u_dict = {'c_crossattn': "", 'c_concat': image_sequence}
        #uc = model.get_learned_conditioning(u_dict)
        #uc = model.enc_concat_seq(uc, u_dict, 'c_concat')
        
        #c_dict = {'c_crossattn': prompt, 'c_concat': image_sequence}
        
        model.eval()
        #c = model.get_learned_conditioning(c_dict)
        #print (c['c_crossattn'].shape)
        #print (c['c_crossattn'][0])
        print (prompt)
        # reshape(B, L * C, H, W)
        #height, width, channels = image_sequence.shape
        # use einsum to reshape
        image_sequence = torch.einsum('hwc->chw', image_sequence).unsqueeze(0)
        c = {'c_concat': image_sequence}
        print (image_sequence.shape, c['c_concat'].shape)
        #c = model.enc_concat_seq(c, c_dict, 'c_concat')
        # Zero out the corresponding subtensors in c_concat for padding images
        #padding_mask = torch.isclose(image_sequence, torch.tensor(-1.0), rtol=1e-5, atol=1e-5).all(dim=(1, 2, 3)).unsqueeze(0)
        #print (padding_mask)
        #padding_mask = padding_mask.repeat(1, 4)  # Repeat mask 4 times for each projected channel
        #print (image_sequence.shape, padding_mask.shape, c['c_concat'].shape)
        #c['c_concat'] = c['c_concat'] * (~padding_mask.unsqueeze(-1).unsqueeze(-1))  # Zero out the corresponding features
        
        
        if pos_maps is not None:
            pos_map = pos_maps[0]
            leftclick_map = torch.cat(leftclick_maps, dim=0)
            print (pos_maps[0].shape, c['c_concat'].shape, leftclick_map.shape)
            if False and DEBUG:
                c['c_concat'] = c['c_concat']*0
            c['c_concat'] = torch.cat([c['c_concat'][:, :, :, :], pos_maps[0].to(c['c_concat'].device).unsqueeze(0), leftclick_map.to(c['c_concat'].device).unsqueeze(0)], dim=1)

        print ('sleeping')
        #time.sleep(120)
        print ('finished sleeping')
        DDPM = False
        DDPM = True
        DDPM = False

        if DEBUG:
            #c['c_concat'] = c['c_concat']*0
            print ('utils prompt', prompt, c['c_concat'].shape, c.keys())
            print (c['c_concat'].nonzero())
            #print (c['c_concat'][0, 0, :, :])

        if DDPM:
            samples_ddim = model.p_sample_loop(cond=c, shape=[1, 4, 48, 64], return_intermediates=False, verbose=True)
        else:
            samples_ddim, _ = sampler.sample(S=16,
                                         conditioning=c,
                                         batch_size=1,
                                         shape=[4, 48, 64],
                                         verbose=False)
        #                                 unconditional_guidance_scale=5.0,
        #                                 unconditional_conditioning=uc,
        #                                 eta=0)

        print ('dfsf1')
        if False and DEBUG:
            print ('samples_ddim.shape', samples_ddim.shape)
            x_samples_ddim = samples_ddim[:, :3]
            # upsample to 512 x 384
            x_samples_ddim = torch.nn.functional.interpolate(x_samples_ddim, size=(384, 512), mode='bilinear')
            # create a 512 x 384 image and paste the samples_ddim into the center
            #x_samples_ddim = torch.zeros((1, 3, 384, 512))
            #x_samples_ddim[:, :, 128:128+48, 160:160+64] = samples_ddim[:, :3]
        else:
            print ('dfsf2')
            data_mean = -0.54
            data_std = 6.78
            data_min = -27.681446075439453
            data_max = 30.854148864746094
            x_samples_ddim = samples_ddim
            x_samples_ddim_feedback = x_samples_ddim
            x_samples_ddim = x_samples_ddim * data_std + data_mean
            x_samples_ddim = model.decode_first_stage(x_samples_ddim)
        print ('dfsf3')
        #x_samples_ddim = pos_map.to(c['c_concat'].device).unsqueeze(0).expand(-1, 3, -1, -1)
        #x_samples_ddim = model.decode_first_stage(x_samples_ddim)
        #x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = torch.clamp(x_samples_ddim, min=-1.0, max=1.0)
        
        return x_samples_ddim.squeeze(0).cpu().numpy(), x_samples_ddim_feedback.squeeze(0)

# Global variables for model and device
#model = None
#device = None

def initialize_model(config_path, model_name):
    #global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_config(config_path, model_name, device)
    return model
