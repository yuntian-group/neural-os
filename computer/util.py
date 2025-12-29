from data.data_processing.datasets import ActionsData
from latent_diffusion.ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch
from typing import List
from data.data_processing.datasets import ActionsData
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train


def _copy_tensor_overlapping(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """
    Copy src into dst on the overlapping top-left slice.

    Requirements:
      - src.ndim == dst.ndim
      - for every dimension: src.shape[d] <= dst.shape[d]
    """
    if src.ndim != dst.ndim:
        raise ValueError(f"ndim mismatch: src.ndim={src.ndim} dst.ndim={dst.ndim}")
    if any(s > d for s, d in zip(src.shape, dst.shape)):
        raise ValueError(f"shape not compatible: src.shape={tuple(src.shape)} dst.shape={tuple(dst.shape)}")
    # keep dst's existing init for the "new" region; overwrite only the overlap
    out = dst.clone()
    slices = tuple(slice(0, int(s)) for s in src.shape)
    out[slices] = src.to(dtype=dst.dtype)
    return out


def _build_partial_unet_state_dict(
    model_state: dict,
    ckpt_state: dict,
    prefix: str = "model.diffusion_model.",
    verbose: bool = True,
) -> dict:
    """
    Build a state_dict to load into the model, allowing partial loads for UNet tensors.

    For keys under `prefix`:
      - if shapes match: load as-is
      - if ckpt tensor is smaller (elementwise per-dim): copy into model tensor
      - otherwise: skip

    For keys outside `prefix`: only load if shapes match exactly.
    """
    out_sd = {}
    exact_loaded = 0
    partial_loaded = 0
    skipped = 0

    for k, v in ckpt_state.items():
        if k not in model_state:
            continue
        dst = model_state[k]
        if not torch.is_tensor(v) or not torch.is_tensor(dst):
            continue

        if k.startswith(prefix):
            if v.shape == dst.shape:
                out_sd[k] = v.to(dtype=dst.dtype)
                exact_loaded += 1
            elif v.ndim == dst.ndim and all(int(s) <= int(d) for s, d in zip(v.shape, dst.shape)):
                out_sd[k] = _copy_tensor_overlapping(dst, v)
                partial_loaded += 1
            else:
                assert False, f"Shape mismatch: {k} {v.shape} {dst.shape}"
        else:
            if v.shape == dst.shape:
                out_sd[k] = v.to(dtype=dst.dtype)
                exact_loaded += 1
            else:
                assert False, f"Shape mismatch: {k} {v.shape} {dst.shape}"

    if verbose:
        print(
            f"[load_model_from_config] prepared state_dict: "
            f"{exact_loaded} exact, {partial_loaded} partial (prefix={prefix}), {skipped} skipped"
        )
    return out_sd


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
    print ('DISABLED training for the autoencoder')
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

    #model.cuda()
    # model.initialize_weights_diffusion_model()
    return model

def load_model_from_config(config, ckpt, verbose=True):

    """
    Loads a pretrained model from config and ckpt.
    """

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model_state = model.state_dict()
    sd = _build_partial_unet_state_dict(
        model_state=model_state,
        ckpt_state=sd,
        prefix="model.diffusion_model.",
        verbose=verbose,
    )
    print(len(sd), 'filtered/partial state dict')
    sd = {
        k: v for k, v in sd.items()
        if k in model_state and v.shape == model_state[k].shape #and ('temporal_encoder' not in k)
    }
    ## remove keys that starts with model.
    #print (f'removing keys that starts with model.')
    #sd = {k: v for k, v in sd.items() if not k.startswith('model.')}
    print (len(sd), 'filtered state dict')
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

    #model.cuda()
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
    print ('loaded first stage')

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
