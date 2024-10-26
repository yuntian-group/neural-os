from data.data_processing.datasets import ActionsData
from latent_diffusion.ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch
from typing import List
from data.data_processing.datasets import ActionsData
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train


def init_model(config: OmegaConf):

    """
    Instantiates the model from a config but doesnt load any weights.
    """

    print("\u23F3 Loading configuration...")
    model = instantiate_from_config(config.model)
    return model


#Loads cond model from ckpt (grabbed from txt2img)
def load_cond_from_ckpt(model: LatentDiffusion, ckpt: str, verbose=False) -> LatentDiffusion:
    
    """
    Loads the actions encoder weights from ckpt for a given model.
    """

    print(f"\u23F3 Loading actions embedding model from {ckpt}")
    m, u = model.cond_stage_model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    return model

#Loads choosen autoencoder from ckpt
def load_autoencoder_from_ckpt(model: LatentDiffusion, ckpt: str, verbose=False) -> LatentDiffusion:

    """
    Loads the autoencoder weights from ckpt for a given model.
    """

    print(f"\u23F3 Loading autoencoder model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location='cpu')
    sd = pl_sd["state_dict"]
    m, u = model.first_stage_model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    #Disables training for the autoencoder
    model.first_stage_model.eval()
    model.first_stage_model.train = disabled_train
    for param in model.first_stage_model.parameters():
        param.requires_grad = False

    return model


def load_model(config):
    """
    Loads the model configuration but no weights. Used to train from scratch. 
    Also inits weights of just the unet. Must load autoencoder and cond after.
    """
    print("\u23F3 Loading configuration...")
    model = instantiate_from_config(config.model)

    model.cuda()
    # model.initialize_weights_diffusion_model()
    return model

def load_model_from_config(config, ckpt, verbose=False):

    """
    Loads a pretrained model from config and ckpt.
    """

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location='cpu')
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    try:
        m, u = model.load_state_dict(sd, strict=False)

        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    except RuntimeError as e:
        # Check if the error message contains "size mismatch"
        if ("size mismatch" and 'input_blocks') in str(e):
            print("\u2757 Input block module could not load weights from ckpt due to the change in input channels.")
        else: raise

    model.cuda()
    model.eval()
    return model

#Loads cond model from ckpt (grabbed from txt2img)
def load_cond_from_config(model: LatentDiffusion, ckpt, verbose=False) -> LatentDiffusion:
    print(f"\u23F3 Loading model from {ckpt}")
    m, u = model.cond_stage_model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    return model

#Loads choosen autoencoder from ckpt
def load_first_stage_from_config(model: LatentDiffusion, ckpt, verbose=False) -> LatentDiffusion:
    print(f"\u23F3 Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location='cpu')
    sd = pl_sd["state_dict"]
    m, u = model.first_stage_model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    #Disables training for the autoencoder
    model.first_stage_model.eval()
    model.first_stage_model.train = disabled_train
    for param in model.first_stage_model.parameters():
        param.requires_grad = False

    return model

def get_ground_truths(data: ActionsData, idxs: List[int]):

    """
    Gets the action and image sequences up to the sequence length, belonging to each target idx.
    """

    prompt_key = 'caption'
    image_seq_cond_key = 'c_concat'
    truth_key = 'image'

    prompts = [data.__getitem__(target_idx)[prompt_key] for target_idx in idxs]
    image_sequences = [data.__getitem__(target_idx)[image_seq_cond_key] for target_idx in idxs]
    target_images = [data.__getitem__(target_idx)[truth_key] for target_idx in idxs]

    return prompts, image_sequences, target_images
