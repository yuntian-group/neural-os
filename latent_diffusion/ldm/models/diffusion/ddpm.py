"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import os
import sys
import torch
import random
import torch.nn as nn
import numpy as np
import lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
#from pytorch_lightning.utilities.distributed import rank_zero_only
import re
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json

from latent_diffusion.ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from latent_diffusion.ldm.modules.ema import LitEma
from latent_diffusion.ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from latent_diffusion.ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from latent_diffusion.ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from latent_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from latent_diffusion.ldm.modules.encoders.temporal_encoder import TemporalEncoder

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=None,  # Changed from fixed size to None
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # Can now be None or a tuple (H, W)
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        if conditioning_key == 'hybrid': print(f"\U0001F9EA Running {conditioning_key}")
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):

        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        print(f"[shared_step] step={self.global_step}")
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        print(f"[training_step] step={self.global_step}")
        DEBUG = True
        DEBUG = False
        self.DEBUG = DEBUG
        if DEBUG:
            print ('no grad at all')
            with torch.no_grad():
                loss, loss_dict = self.shared_step(batch)
        else:
            loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True, sync_dist=True)

        #self.log("global_step", self.global_step,
        #         prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)

        if self.use_scheduler:
            assert False
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 temporal_encoder_config=None,  # New parameter
                 num_timesteps_cond=None,
                 scheduler_sampling_rate=0,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 hybrid_key=None, #for csllm
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scheduler_sampling_rate = scheduler_sampling_rate
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        #if cond_stage_config == '__is_unconditional__':
        #    conditioning_key = None
        self.cond_stage_config = cond_stage_config
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.hybrid_key = hybrid_key #key for hybrid conditioning
        # assert hybrid_key, conditioning_key in ['c_concat', 'caption', 'class_label', None] #only combinations that will work
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.temporal_encoder = None
        if temporal_encoder_config is not None:
            self.temporal_encoder = instantiate_from_config(temporal_encoder_config)
        DEBUG = True
        if DEBUG:
            from torchvision import transforms
            from PIL import Image
            transform = transforms.ToTensor()
            print("Loading cluster centers...")
            # Define cluster directories
            cluster_dir = "filtered_transition_clusters"
            cluster_paths = sorted(Path(cluster_dir).glob("cluster_*_size_*"))
            cluster_centers = []
            cluster_ids = []
            for cluster_path in cluster_paths:
                if "noise" in str(cluster_path):
                    continue
                    
                # Extract cluster ID using regex (e.g., "cluster_5_size_100" -> 5)
                match = re.search(r'cluster_(\d+)_size_', str(cluster_path.name))
                if not match:
                    continue
                cluster_id = int(match.group(1))
                
                # Find center images
                center_files = list(cluster_path.glob("cluster_center_*.png"))
                if not center_files:
                    continue
                    
                # Load and concatenate center images
                prev_img = transform(Image.open([f for f in center_files if 'prev' in str(f)][0])).view(-1)
                curr_img = transform(Image.open([f for f in center_files if 'curr' in str(f)][0])).view(-1)
                center = torch.cat([prev_img, curr_img], dim=0)
                
                cluster_centers.append(center)
                cluster_ids.append(cluster_id)
            cluster_centers = torch.stack(cluster_centers)
            print (cluster_centers)
            self.cluster_centers = cluster_centers
            self.cluster_ids = cluster_ids

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    ###@rank_zero_only
    ###@torch.no_grad()
    ###def on_train_batch_start(self, batch, batch_idx): #, dataloader_idx):
    ###    # only for very first batch
    ###    if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
    ###        assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
    ###        # set rescale weight to 1./std of encodings
    ###        print("### USING STD-RESCALING ###")
    ###        x = super().get_input(batch, self.first_stage_key)
    ###        x = x.to(self.device)
    ###        encoder_posterior = self.encode_first_stage(x)
    ###        z = self.get_first_stage_encoding(encoder_posterior).detach()
    ###        del self.scale_factor
    ###        self.register_buffer('scale_factor', 1. / z.flatten().std())
    ###        print(f"setting self.scale_factor to {self.scale_factor}")
    ###        print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                #If hybrid, recreate the condition dict
                if isinstance(c, dict):
                    enc_c = self.cond_stage_model.encode(c['c_crossattn'])
                    c['c_crossattn'] = enc_c #update encoded cond to the cond dict.
                else:
                    c = self.cond_stage_model.encode(c) #used in bert tokenizer
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def enc_concat_seq(self, c: dict, batch, k) -> tuple:
        """
        encodes a sequence of images from a batch for conditioning. used with c_concat and hybrid conditioning.
        Will use preprocessed latents if available, otherwise encodes images.
        Returns:
            tuple: (updated condition dictionary, padding mask)
        """
        assert k == 'c_concat', "Only concat conditioning is supported for now"
        
        # If preprocessed latents are available, use them directly
        if f'{k}_processed' in batch:
            image_sequence = batch[f'{k}_processed']  # shape: [B, L, C, H, W]
            # Create padding mask based on preprocessed latents
            # For preprocessed data, padding is all zeros
            is_padding = (image_sequence.abs() < 1e-6).all(dim=(2,3,4))  # shape: [batch_size, sequence_length]
            
            # Reshape to combine sequence length and channel dimensions
            B, L, C, H, W = image_sequence.shape
            c[k] = image_sequence.reshape(B, L * C, H, W)
            return c, is_padding
            
        # Otherwise, encode the images
        enc_sequence = []
        image_sequence = batch[k]

        if len(image_sequence.shape) == 4:
            image_sequence = image_sequence.unsqueeze(0) #add batch dimension if missing

        input = rearrange(image_sequence, 'b l h w c -> l b c h w')

        # Create padding mask based on input sequence
        # Check if all values are close to -1 (with epsilon for floating point precision)
        eps = 1e-6
        is_padding = ((image_sequence + 1).abs() < eps).all(dim=(2,3,4))  # shape: [batch_size, sequence_length]

        #we need to encode each image in the sequence l before concat.
        for img in input:
            img = img.to(self.device) # b c h w
            img = img.to(memory_format=torch.contiguous_format).float()
            img = self.encode_first_stage(img)
            enc_sequence.append(img)

        c[k] = torch.cat(enc_sequence, dim=1) #concat all the latents together on the channel dim
        return c, is_padding

        
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):

        
        assert k == 'image', "Only image conditioning is supported for now"
        #import pdb; pdb.set_trace()

        if 'image_processed' in batch:
            z = batch['image_processed']
            assert bs is None, "Batch size must be None when using processed images"
            z = z.to(self.device)
        else:
            x = super().get_input(batch, k)
            if bs is not None:
                x = x[:bs]
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
        
        debug = False
        if debug:
            import pdb; pdb.set_trace()
            from PIL import Image, ImageDraw
            #image = Image.fromarray(z[-1].transpose(0, 1).transpose(1, 2).cpu().numpy())
            for kkk in range(32):
                image = Image.fromarray(((z[kkk].transpose(0, 1).transpose(1, 2).cpu().float().numpy()+1)*255/2).astype(np.uint8))
                a = batch['action_7'][kkk].replace(' ','').replace(':', '').split('+')
                x, y = int(a[-2]), int(a[-1])
                x_scaled = int(x / 1024 * 64)
                y_scaled = int(y / 640 * 64)
                draw = ImageDraw.Draw(image)
                draw.ellipse([x_scaled-2, y_scaled-2, x_scaled+2, y_scaled+2], fill=(255, 0, 0))
                image.save(f'loadmodel64debug_{kkk}.png')
            import pdb; pdb.set_trace()

        self.context_length = self.trainer.datamodule.datasets['train'].context_length

        if (self.model.conditioning_key is not None) and (self.cond_stage_config != '__is_unconditional__'):
            if cond_key is None:
                cond_key = self.cond_stage_key
            cond_key = f'action_{self.context_length}'
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox', f'action_{self.context_length}']:
                    xc = batch[cond_key]

                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}

        #This is for concat + cross attention conditioning only, indicated with a hybrid key.
        if self.hybrid_key == 'c_concat':
            #import pdb; pdb.set_trace()
            hkey = self.hybrid_key

            c = {'c_crossattn': batch[cond_key]} if cond_key is not None else {} #cond_key is converted to cross attention.
            c, is_padding = self.enc_concat_seq(c, batch, hkey)
            #return the dict of conds with cattn for the learnable cond. and cconcat for latent cond.
            if self.temporal_encoder is not None:
                #assert False, "Temporal encoder is not implemented"
                #assert f'{hkey}_processed' in batch, "Processed sequence is required for temporal encoder"
                #import pdb; pdb.set_trace()
                # c_concat_processed: previous image sequences, shape: [B, L, C, H, W], with L being 14.
                # position_map_j: position map for the jth frame, shape: [B, 1, H, W], with j being from -7 to 7 (15 in total)
                # leftclick_map_j: leftclick map for the jth frame, shape: [B, 1, H, W], with j being from -7 to 7 (15 in total)
                # concatenate through the channel dimension.
                #history_length_to_consider = 14
                #actual_history_length = batch['c_concat_processed'].shape[1]

                inputs_to_rnn = []
                #TRIM_BEGINNING = 1
                #TRIM_BEGINNING = 1

                for t in range(self.context_length):
                    inputs_t = {}
                    inputs_t['image_features'] = batch[f'c_concat_processed'][:, self.context_length + t]
                    inputs_t['is_padding'] = batch[f'is_padding'][:, self.context_length + t] # if is_padding is true, then set initial state to the padding state
                    inputs_t['x'] = batch[f'x_{t+1}']
                    inputs_t['y'] = batch[f'y_{t+1}']
                    inputs_t['is_leftclick'] = batch[f'is_leftclick_{t+1}']
                    inputs_t['is_rightclick'] = batch[f'is_rightclick_{t+1}']
                    inputs_t['key_events'] = batch[f'key_events_{t+1}']
                    inputs_to_rnn.append(inputs_t)
                #for j in range(history_length_to_consider):
                #    image_part = batch[f'c_concat_processed'][:, actual_history_length-history_length_to_consider+j]
                #    position_map_part = batch[f'position_map_{7-history_length_to_consider+j+TRIM_BEGINNING}']
                #    leftclick_map_part = batch[f'leftclick_map_{7-history_length_to_consider+j+TRIM_BEGINNING}']
                #    inputs_to_rnn.append(torch.cat([image_part, position_map_part, leftclick_map_part], dim=1))
                #inputs_to_rnn = torch.stack(inputs_to_rnn, dim=1)
                #import pdb; pdb.set_trace()
                with torch.enable_grad():
                    output_from_rnn = self.temporal_encoder(inputs_to_rnn)
                ###output_from_rnn = self.temporal_encoder(inputs_to_rnn)
                ###print ('warning: no grad')
                
                #output_from_rnn = self.temporal_encoder(inputs_to_rnn)
                #import pdb; pdb.set_trace()
                #c[hkey] = c[hkey][:, 4*7:]
                c[hkey] = output_from_rnn
                #if TRIM_BEGINNING == 0:
                #    pos_map = batch['position_map_7']
                #    inputs = [c[hkey], pos_map] + [batch[f'leftclick_map_7']]
                #    c[hkey] = torch.cat(inputs, dim=1)
                # concatenate with the position map.
                #assert c[hkey].shape[1] == 4*7 + 2 + 7
            else:
                assert False
                #c[hkey] = output_from_rnn

                #c, is_padding = self.enc_concat_seq(c, batch, hkey)
                data_mean = -0.54
                data_std = 6.78
                data_min = -27.681446075439453
                data_max = 30.854148864746094
                proposal = random.random() 
                if proposal < self.scheduler_sampling_rate:
                    #assert False, "Not implemented"
                    #import pdb; pdb.set_trace()
                    with torch.no_grad():
                        
                        #assert cond_key == 'action_7', "Only action conditioning is supported for now"
                        scheduled_sampling_length = 1
                        while scheduled_sampling_length < self.context_length:
                            if random.random() < 0.5:
                                scheduled_sampling_length += 1
                            else:
                                break
                        print ('='*100)
                        print (f"Scheduled sampling length: {scheduled_sampling_length}")
                        #for j in range(self.context_length-1, -1, -1):
                        for j in range(self.context_length-scheduled_sampling_length, self.context_length):
                            c_prev = c[hkey][:, 4*j:4*(j+self.context_length)]
                            c_dict = {'c_concat': c_prev}
                            #c_dict = self.get_learned_conditioning(c_dict)
                            batch_size = c_prev.shape[0]
                            #uc_dict = {'c_crossattn': ['']*batch_size, 'c_concat': c_prev}
                            #uc_dict = self.get_learned_conditioning(uc_dict)
                            sampler = DDIMSampler(self)
                            position_map = batch[f'position_map_{j}']
                            c_dict['c_concat'] = torch.cat([c_dict['c_concat'], position_map] + [batch[f'leftclick_map_{k+j-self.context_length}'] for k in range(self.context_length, -1, -1)], dim=1)
                            #uc_dict['c_concat'] = torch.cat([uc_dict['c_concat'], position_map], dim=1)
                            samples_ddim, _ = sampler.sample(S=8,
                                            conditioning=c_dict,
                                            batch_size=batch_size,
                                            shape=[4, 48, 64],
                                            verbose=False,)
                                            #unconditional_guidance_scale=5.0,
                                            #unconditional_conditioning=uc_dict,
                                            #eta=0)
                            #samples_ddim = samples_ddim * data_std + data_mean
                            # Decode in smaller batches
                            decode_batch_size = batch_size #16
                            x_samples_ddim = []
                            z_samples = []
                            for idx in range(0, samples_ddim.shape[0], decode_batch_size):
                                batch_samples = samples_ddim[idx:min(idx + decode_batch_size, samples_ddim.shape[0])]
                                #batch_decoded = self.decode_first_stage(batch_samples)
                                #batch_encoded = torch.clamp(batch_decoded, min=-1.0, max=1.0)
                                #x_samples_ddim.append(batch_decoded)
                                #batch_encoded = self.encode_first_stage(batch_decoded).sample()
                                #z_samples.append(batch_encoded)
                                z_samples.append(batch_samples)
                            #x_samples_ddim = torch.cat(x_samples_ddim, dim=0)
                            #x_samples_ddim = torch.clamp(x_samples_ddim, min=-1.0, max=1.0)
                            z_samples = torch.cat(z_samples, dim=0)
                            #z_samples = (z_samples - data_mean) / data_std # only use normalization when encoding again
                            # save to disk for visualization and debugging
                            #for kkk in range(batch_size):
                            #       from PIL import Image, ImageDraw
                            #       image = Image.fromarray(((x_samples_ddim[kkk].transpose(0, 1).transpose(1, 2).cpu().float().numpy()+1)*255/2).astype(np.uint8))
                            #       image.save(f'feb15_25_ddim_sample_{j}_{kkk}.png')
                            #
                            #import pdb; pdb.set_trace()
                            # Encode the generated samples back to latent space
                            #z_samples = self.encode_first_stage(x_samples_ddim)
                            
                            # Replace the corresponding frames in c[hkey]
                            sampling_mask = torch.rand(batch_size, 1, 1, 1, device=c[hkey].device) < 1.5 #self.scheduler_sampling_rate
                            # Only apply sampling mask where is_padding is False
                            mask = sampling_mask & (~is_padding[:, j+self.context_length].view(-1, 1, 1, 1))
                            #if is_padding[:, j+7].any():
                            #    import pdb; pdb.set_trace()
                            c[hkey][:, self.context_length*4+j*4:self.context_length*4+j*4+4] = torch.where(mask, z_samples, c[hkey][:, self.context_length*4+j*4:self.context_length*4+j*4+4])
                            #break
                elif proposal < self.scheduler_sampling_rate + 0.1:
                    #assert False, "Not implemented"
                    #import pdb; pdb.set_trace()
                    with torch.no_grad():
                        
                        #assert cond_key == 'action_7', "Only action conditioning is supported for now"
                        #scheduled_sampling_length = 1
                        #while scheduled_sampling_length < self.context_length:
                        #    if random.random() < 0.5:
                        #        scheduled_sampling_length += 1
                        #    else:
                        #        break
                        print ('*'*100)
                        #print (f"Scheduled sampling length: {scheduled_sampling_length}")
                        for j in range(self.context_length-1, -1, -1):
                        #for j in range(self.context_length-scheduled_sampling_length, self.context_length):
                            c_prev = c[hkey][:, 4*j:4*(j+self.context_length)]
                            c_dict = {'c_concat': c_prev}
                            #c_dict = self.get_learned_conditioning(c_dict)
                            batch_size = c_prev.shape[0]
                            #uc_dict = {'c_crossattn': ['']*batch_size, 'c_concat': c_prev}
                            #uc_dict = self.get_learned_conditioning(uc_dict)
                            sampler = DDIMSampler(self)
                            position_map = batch[f'position_map_{j}']
                            c_dict['c_concat'] = torch.cat([c_dict['c_concat'], position_map] + [batch[f'leftclick_map_{k+j-self.context_length}'] for k in range(self.context_length, -1, -1)], dim=1)
                            #uc_dict['c_concat'] = torch.cat([uc_dict['c_concat'], position_map], dim=1)
                            samples_ddim, _ = sampler.sample(S=8,
                                            conditioning=c_dict,
                                            batch_size=batch_size,
                                            shape=[4, 48, 64],
                                            verbose=False,)
                                            #unconditional_guidance_scale=5.0,
                                            #unconditional_conditioning=uc_dict,
                                            #eta=0)
                            #samples_ddim = samples_ddim * data_std + data_mean
                            # Decode in smaller batches
                            decode_batch_size = batch_size #16
                            x_samples_ddim = []
                            z_samples = []
                            for idx in range(0, samples_ddim.shape[0], decode_batch_size):
                                batch_samples = samples_ddim[idx:min(idx + decode_batch_size, samples_ddim.shape[0])]
                                #batch_decoded = self.decode_first_stage(batch_samples)
                                #batch_encoded = torch.clamp(batch_decoded, min=-1.0, max=1.0)
                                #x_samples_ddim.append(batch_decoded)
                                #batch_encoded = self.encode_first_stage(batch_decoded).sample()
                                #z_samples.append(batch_encoded)
                                z_samples.append(batch_samples)
                            #x_samples_ddim = torch.cat(x_samples_ddim, dim=0)
                            #x_samples_ddim = torch.clamp(x_samples_ddim, min=-1.0, max=1.0)
                            z_samples = torch.cat(z_samples, dim=0)
                            #z_samples = (z_samples - data_mean) / data_std # only use normalization when encoding again
                            # save to disk for visualization and debugging
                            #for kkk in range(batch_size):
                            #       from PIL import Image, ImageDraw
                            #       image = Image.fromarray(((x_samples_ddim[kkk].transpose(0, 1).transpose(1, 2).cpu().float().numpy()+1)*255/2).astype(np.uint8))
                            #       image.save(f'feb15_25_ddim_sample_{j}_{kkk}.png')
                            #
                            #import pdb; pdb.set_trace()
                            # Encode the generated samples back to latent space
                            #z_samples = self.encode_first_stage(x_samples_ddim)
                            
                            # Replace the corresponding frames in c[hkey]
                            sampling_mask = torch.rand(batch_size, 1, 1, 1, device=c[hkey].device) < 1.5 #self.scheduler_sampling_rate
                            # Only apply sampling mask where is_padding is False
                            mask = sampling_mask & (~is_padding[:, j+self.context_length].view(-1, 1, 1, 1))
                            #if is_padding[:, j+7].any():
                            #    import pdb; pdb.set_trace()
                            c[hkey][:, self.context_length*4+j*4:self.context_length*4+j*4+4] = torch.where(mask, z_samples, c[hkey][:, self.context_length*4+j*4:self.context_length*4+j*4+4])
                            #break

                c[hkey] = c[hkey][:, 4*self.context_length:] #* 0 # TODO: remove
                #import pdb; pdb.set_trace()
                pos_map = batch[f'position_map_{self.context_length}']
                #leftclick_map = batch['leftclick_map_7']
                #c[hkey] = torch.cat([c[hkey], pos_map], dim=1)
                #c[hkey] = torch.cat([c[hkey], pos_map, leftclick_map], dim=1)
                inputs = [c[hkey], pos_map] + [batch[f'leftclick_map_{j}'] for j in range(self.context_length, -1, -1)]
                #import pdb; pdb.set_trace()
                c[hkey] = torch.cat(inputs, dim=1)
                # concatenate with the position map.
                assert c[hkey].shape[1] == 4*self.context_length + 2 + self.context_length
        else:
            assert False, "Only concat conditioning is supported for now"

        DEBUG = True
        DEBUG = False
        exp_name = 'without'
        exp_name = '8192_1layer'
        exp_name = '8192_1layer_trim'

        exp_name = 'without_comp_norm_minmax'
        exp_name = 'without_comp_norm_none'
        exp_name = 'without_comp_norm_standard'
        DDIM_S = 8
        DEBUG = False
        #### REPLACEMENT_LINE
        os.makedirs(exp_name, exist_ok=True)
        if not hasattr(self, 'i'):
            self.i = 0

        if DEBUG:
            device = c[hkey].device
            self.cluster_centers = self.cluster_centers.to(device)
            cluster_centers = self.cluster_centers
            cluster_ids = self.cluster_ids
            
            print(f"Loaded {len(cluster_centers)} cluster centers")
            #data_mean = -0.54
            #data_std = 6.78
            #data_min = -27.681446075439453
            #data_max = 30.854148864746094
            per_channel_mean = self.trainer.datamodule.datasets['train'].per_channel_mean.to(device)
            per_channel_std = self.trainer.datamodule.datasets['train'].per_channel_std.to(device)
            # Define icon boundaries
            ICONS = {
                    'firefox': {'center': (66, 332-30), 'width': int(22*1.4), 'height': 44},
                    'root': {'center': (66, 185), 'width': int(22*1.95), 'height': 42},
                    'terminal': {'center': (191, 60), 'width': int(22*2), 'height': 44},
                    'trash': {'center': (66, 60), 'width': int(22*1.95), 'height': 42}
                }
            self.eval()
            #import pdb; pdb.set_trace()
            if 'norm_standard' in exp_name:
                batch['image_processed'] = batch['image_processed'] * per_channel_std.view(1, -1, 1, 1) + per_channel_mean.view(1, -1, 1, 1)
                batch['c_concat_processed'] = batch['c_concat_processed'] * per_channel_std.view(1, 1, -1, 1, 1) + per_channel_mean.view(1, 1, -1, 1, 1)
            else:
                assert False
            #elif 'without_comp_norm_minmax' in exp_name:
            #    batch['image_processed'] = (batch['image_processed'].clamp(-1, 1) + 1) * (data_max - data_min) / 2 + data_min
            #    batch['c_concat_processed'] = (batch['c_concat_processed'].clamp(-1, 1) + 1) * (data_max - data_min) / 2 + data_min
            #else:
            #    pass
            z_vis = self.decode_first_stage(batch['image_processed'])
            #prev_frames = self.decode_first_stage(batch['c_concat_processed'][:, -1])
            c['c_concat'] = c['c_concat'].data.clone()
            for i, zz in enumerate(z_vis):
                prev_frames = self.decode_first_stage(batch['c_concat_processed'][i, -self.context_length:])
                prev_frames = prev_frames.clamp(-1, 1)
                #prev_frame = prev_frames[i].clamp(-1, 1)
                from PIL import Image
                import copy
                #Image.fromarray(((zz.transpose(0,1).transpose(1,2).cpu().float().numpy()+1)*255/2).astype(np.uint8)).save(f'leftclick_debug_image_{i}.png')
                
                c_i = copy.deepcopy(c)
                c_i['c_concat'] = c['c_concat'][i:i+1]
                #c_i['c_crossattn'] = c['c_crossattn'][i:i+1]
                #c_i = self.get_learned_conditioning(c_i)
                ddpm = False
                if ddpm:
                    sample_i = self.p_sample_loop(cond=c_i, shape=[1, 16, 48, 64], return_intermediates=False, verbose=True)
                else:
                    print ('ddim', DDIM_S)
                    sampler = DDIMSampler(self)
                    sample_i , _ = sampler.sample(
                        S=DDIM_S,
                        conditioning=c_i,
                        batch_size=1,
                        shape=[16, 48, 64],
                        verbose=False
                    )
                
                if 'norm_standard' in exp_name:
                    sample_i = sample_i * per_channel_std.view(1, -1, 1, 1) + per_channel_mean.view(1, -1, 1, 1)
                    #prev_frame_img = prev_frame_img * data_std + data_mean
                else:
                    assert False
                sample_i = self.decode_first_stage(sample_i)
                sample_i = sample_i.squeeze(0).clamp(-1, 1)
                # plot sample_i side by side with zz
                # Convert tensors to numpy arrays and prepare for visualization
                zz = zz.clamp(-1, 1)
                zz_img = ((zz.transpose(0,1).transpose(1,2).cpu().float().numpy() + 1) * 127.5).astype(np.uint8)
                sample_img = ((sample_i[:3].transpose(0,1).transpose(1,2).cpu().float().numpy() + 1) * 127.5).astype(np.uint8)
                
                # Create a new image with twice the width to hold both images
                #combined_img = np.zeros((48*8, 64*8*3, 3), dtype=np.uint8)
                #combined_img = np.zeros((48*8, 64*8*2, 3), dtype=np.uint8)
                #combined_img[:, :64*8] = prev_frame_img  # Original on left
                #combined_img[:, 64*8:64*8*2] = zz_img  # Generated on right
                # combined_img[:, 64*8*2:] = sample_img  # Generated on right
                
                # Save the combined image
                #Image.fromarray(combined_img).save(f'{exp_name}/real_vs_generated_debug_comparison_{self.i}.png')
                
                # Save the corresponding action texts
                #with open(f'{exp_name}/real_vs_generated_debug_comparison_{self.i}.txt', 'w') as f:
                #    action_7 = batch[f'action_{self.context_length}'][i]
                #    action_0 = batch[f'action_0'][i]
                #    f.write(f"Current action (7): {action_7}\n")
                #    f.write(f"First action (0): {action_0}\n")

                
                # Initialize confusion matrix if not exists
                if not hasattr(self, 'confusion_matrix'):
                    #cluster_names = list(cluster_paths.keys())
                    cluster_names = [str(cluster_id) for cluster_id in cluster_ids]
                    self.confusion_matrix = np.zeros((len(cluster_ids), len(cluster_ids)), dtype=int)
                    self.cluster_names = cluster_names
                
                # Function to find closest cluster
                def get_closest_cluster(prev_img, curr_img):
                    #min_mse = float('inf')
                    #closest_name = None
                    img_prev = (prev_img + 1) * 127.5 / 255
                    img_curr = (curr_img + 1) * 127.5 / 255
                    img_prev = (img_prev).to(device).reshape(-1)
                    img_curr = (img_curr).to(device).reshape(-1)
                    img = torch.cat([img_prev, img_curr], dim=0).unsqueeze(0)
                    distances = torch.norm(img - cluster_centers, dim=1)
                    min_idx = distances.argmin().item()
                    
                    return cluster_ids[min_idx]
                    
                    #for name, center in cluster_centers.items():
                    #    mse = np.mean((img_np - center) ** 2)
                    #    if mse < min_mse:
                    #        min_mse = mse
                    #        closest_name = name
                    #return closest_name

                # Get cluster assignments
                #import pdb; pdb.set_trace()
                target_idx = get_closest_cluster(prev_frames[-1], zz)
                pred_idx = get_closest_cluster(prev_frames[-1], sample_i)
                
                # Update confusion matrix
                #target_idx = self.cluster_names.index(target_cluster)
                #pred_idx = self.cluster_names.index(pred_cluster)
                self.confusion_matrix[target_idx][pred_idx] += 1
                
                # Create directory for this pair
                pair_dir = f'{exp_name}/target_{target_idx}_pred_{pred_idx}'
                os.makedirs(pair_dir, exist_ok=True)

                # [Keep existing visualization code here]
                # Parse action sequence
                def parse_action_sequence(action_str):
                    # Remove all spaces
                    action_str = action_str.replace(" ", "")
                    # Use regex to find all actions
                    actions = re.findall(r'([NL])\+(\d+):\+(\d+)', action_str)
                    return [(action_type, int(x), int(y)) for action_type, x, y in actions]

                #actions = parse_action_sequence(action_7)
                is_leftclicks = [batch[f'is_leftclick_{j}'][i] for j in range(self.context_length+1)]
                is_rightclicks = [batch[f'is_rightclick_{j}'][i] for j in range(self.context_length+1)]
                key_events = [batch[f'key_events_{j}'][i] for j in range(self.context_length+1)]
                xs = [batch[f'x_{j}'][i].item() for j in range(self.context_length+1)]
                ys = [batch[f'y_{j}'][i].item() for j in range(self.context_length+1)]
                
                #assert len(actions) == self.context_length+1, (action_7, actions)
                #actions = actions[-7:]
                
                # Calculate grid dimensions (roughly 2:1 aspect ratio)
                total_images = self.context_length + 2 + 1  # 7 history frames + target + prediction
                cols = int(np.sqrt(total_images * 2))  # multiply by 2 for 2:1 aspect ratio
                rows = (total_images + cols - 1) // cols  # ceiling division
                
                # Create combined image with proper dimensions
                frame_height, frame_width = 48*8, 64*8
                combined_img = np.zeros((frame_height * rows, frame_width * cols, 3), dtype=np.uint8)
                
                def draw_action_on_frame(img, is_leftclick, is_rightclick, key_events, x, y):
                    img = img.copy()
                    for name, icon in ICONS.items():
                        center = (int(icon['center'][0]), int(icon['center'][1]))
                        width = int(icon['width'])
                        height = int(icon['height'])
                        
                        
                        x1 = center[0] - width
                        y1 = center[1] - height
                        x2 = center[0] + width
                        y2 = center[1] + height
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow rectangle
                    
                    
                    if is_leftclick:
                        # Red circle for clicks
                        inside_icon = False
                        for name, icon in ICONS.items():
                            center = (int(icon['center'][0]), int(icon['center'][1]))
                            width = int(icon['width'])
                            height = int(icon['height'])
                            if (abs(x - center[0]) <= width and 
                                abs(y - center[1]) <= height):
                                inside_icon = True
                                break
                        
                        # Green circle for clicks inside icons, red for clicks outside
                        color = (0, 255, 0) if inside_icon else (255, 0, 0)
                        cv2.circle(img, (x, y), 10, color, 3)
                    elif is_rightclick:
                        # Blue circle for right clicks
                        cv2.circle(img, (x, y), 10, (0, 0, 255), 3)
                    else:
                        # White dot for moves
                        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
                    # write key downs on the frame
                    texts = []
                    for key_id, is_down in enumerate(key_events):
                        if is_down:
                            key_name = self.trainer.datamodule.datasets['train'].itos[key_id]
                            texts.append(key_name)
                    # write all texts on the center of the frame, one per line
                    # Draw each key on a separate line
                    font_scale = 0.5
                    thickness = 2
                    line_spacing = 5
                    
                    # Get initial text height using a sample text
                    (_, text_height), _ = cv2.getTextSize("Sample", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    
                    # Calculate total height of all text lines
                    total_height = len(texts) * (text_height + line_spacing) - line_spacing
                    
                    # Start from vertical center
                    y_position = (frame_height - total_height) // 2
                    
                    for text in texts:
                        # Calculate text size for centering
                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        text_x = (frame_width - text_width) // 2  # Center horizontally
                        
                        # Draw red text
                        cv2.putText(img, text, (text_x, y_position), 
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
                        
                        y_position += text_height + line_spacing
                        
                    return img

                # Draw all frames in grid
                for j in range(rows * cols):
                    
                    row = j // cols
                    col = j % cols
                    if j == 0:
                        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                        frame = draw_action_on_frame(frame, is_leftclicks[j], is_rightclicks[j], key_events[j], xs[j], ys[j])
                        combined_img[row*frame_height:(row+1)*frame_height, 
                                   col*frame_width:(col+1)*frame_width] = frame
                    elif j < self.context_length+1:  # History frames
                        prev_frame = prev_frames[j-1]
                        prev_frame_img = ((prev_frame.transpose(0,1).transpose(1,2).cpu().float().numpy()+1)*255/2).astype(np.uint8)
                        frame = prev_frame_img
                        #frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                        frame = draw_action_on_frame(frame, is_leftclicks[j], is_rightclicks[j], key_events[j], xs[j], ys[j])
                        combined_img[row*frame_height:(row+1)*frame_height, 
                                   col*frame_width:(col+1)*frame_width] = frame
                    elif j == self.context_length+1:  # Target frame
                        combined_img[row*frame_height:(row+1)*frame_height, 
                                   col*frame_width:(col+1)*frame_width] = zz_img
                    elif j == self.context_length+2:  # Prediction frame
                        combined_img[row*frame_height:(row+1)*frame_height, 
                                   col*frame_width:(col+1)*frame_width] = sample_img

                Image.fromarray(combined_img).save(f'{pair_dir}/comparison_{self.i}.png')
                
                # Save confusion matrix periodically
                # compute accuracy, precision, recall, f1 score
                precision = self.confusion_matrix.diagonal() / np.clip(self.confusion_matrix.sum(axis=1), 1e-5, None)
                precision_mean = precision.mean()
                recall = self.confusion_matrix.diagonal() / np.clip(self.confusion_matrix.sum(axis=0), 1e-5, None)
                recall_mean = recall.mean()
                f1_score = 2 * precision * recall / np.clip(precision + recall, 1e-5, None)
                f1_score_mean = f1_score.mean()
                accuracy = self.confusion_matrix.diagonal().sum() / self.confusion_matrix.sum()
                setting = exp_name
                print (f'setting: {setting}')
                print (f'precision breakdown: {precision}')
                print (f'recall breakdown: {recall}')
                print (f'f1_score breakdown: {f1_score}')
                print (f'precision: {precision_mean}, recall: {recall_mean}, f1_score: {f1_score_mean}, accuracy: {accuracy}')
                print ('='*100)
                # write to a file and compare with previous results
                if os.path.exists('all_psearch_results.json'):
                    try:
                        with open('all_psearch_results.json', 'r') as f:
                            all_results = json.load(f)
                    except:
                        all_results = {}
                else:
                    all_results = {}
                all_results[setting] = {
                    'precision_mean': precision_mean,
                    'recall_mean': recall_mean,
                    'f1_score_mean': f1_score_mean,
                    'precision_breakdown': precision.tolist(),
                    'recall_breakdown': recall.tolist(),
                    'f1_score_breakdown': f1_score.tolist(),
                    'accuracy': accuracy
                }
                with open('all_psearch_results.json', 'w') as f:
                    json.dump(all_results, f)
                # compare with previous results and sort by accuracy
                all_results_sorted = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
                
                # Display sorted results in a more readable format
                print("\n=== RESULTS RANKED BY ACCURACY ===")
                print(f"{'Rank':<5}{'Setting':<40}{'Accuracy':<10}{'F1 Score':<10}{'Precision':<10}{'Recall':<10}")
                print("-" * 85)
                for i, (setting_name, metrics) in enumerate(all_results_sorted):
                    print(f"{i+1:<5}{setting_name:<40}: accuracy: {metrics['accuracy']:.4f}, f1_score: {metrics['f1_score_mean']:.4f}, precision: {metrics['precision_mean']:.4f}, recall: {metrics['recall_mean']:.4f}")
                    print(f"precision breakdown: {metrics['precision_breakdown']}")
                    print(f"recall breakdown: {metrics['recall_breakdown']}")
                    print(f"f1_score breakdown: {metrics['f1_score_breakdown']}")
                print("=" * 85)

                if True or self.i % 10 == 0:
                    plt.figure(figsize=(10,8))
                    sns.heatmap(self.confusion_matrix, 
                               xticklabels=self.cluster_names,
                               yticklabels=self.cluster_names,
                               annot=True, fmt='d')
                    plt.title('Prediction Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('Target')
                    plt.savefig(f'{exp_name}/confusion_matrix.png')
                    plt.close()
                    ## Inside your model code where confusion matrix is calculated
                    #history_length = self.context_length
                    #if 'train' not in exp_name:
                    #    for icon in self.cluster_names:
                    #        i = self.cluster_names.index(icon)
                    #        total = self.confusion_matrix[i].sum()
                    #        if total > 0:
                    #            accuracy = self.confusion_matrix[i,i] / total
                    #            update_accuracy_csv(history_length, icon, accuracy, total)

                self.i += 1
                #if self.i >= 497:
                #    sys.exit(1)
                if self.i >= 150:
                    sys.exit(1)
            #import pdb; pdb.set_trace()


        out = [z, c]
        #import pdb; pdb.set_trace()
        if 'c_crossattn' in c:
            c['c_crossattn'] = [' '.join(['N' for item in items.split()]) for items in c['c_crossattn']] # TODO: note that encoder is not used
        #import pdb; pdb.set_trace()
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):  
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        #import pdb; pdb.set_trace()

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict <---------------
            # print("its hybrid!!!!!")
            pass
        else:
            if not isinstance(cond, list):
                # print("AHAH GOT U")
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            #This is where we concat the noisy latents with the cond images.
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None):
        # print("p losses calledGHGHJGJHGJHGJHGJHGJHGJHGJHGJHGJHG")
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise) 
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        #move t to cpu with logvar
        logvar_t = self.logvar[t.to(self.logvar.device)].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, **kwargs):
        if shape is None and self.image_size is not None:
            # Support tuple image_size
            if isinstance(self.image_size, (tuple, list)):
                shape = (batch_size, self.channels, *self.image_size)
            else:
                shape = (batch_size, self.channels, self.image_size, self.image_size)
        if shape is None:
            raise ValueError("Either shape or self.image_size must be specified")
            
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                              shape,
                              return_intermediates=return_intermediates, x_T=x_T,
                              verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                              mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates


    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def initialize_weights_diffusion_model(self):
        """
        Apply the weight initialization to all modules in the diffusion model.
        """
        for module in self.model.diffusion_model.modules():
            self.init_weights_module(module)
    
    def init_weights_module(self, module):
        """
        Custom weight initializer for modules.
        Args:
            module: A module to initialize.
        """
        if isinstance(module, nn.Conv2d):
            # Kaiming initialization for Conv2d layers
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize biases to zero

        elif isinstance(module, nn.Linear):
            # Xavier initialization for Linear layers
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize biases to zero

        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            # Initialize BatchNorm or GroupNorm layers
            nn.init.ones_(module.weight)  # Set weight to 1
            nn.init.zeros_(module.bias)   # Set bias to 0

        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm layers
            nn.init.ones_(module.weight)  # Set weight to 1
            nn.init.zeros_(module.bias)   # Set bias to 0

        elif isinstance(module, nn.Embedding):
            # Initialize embeddings
            nn.init.normal_(module.weight, mean=0, std=0.01)  # Example initialization for embeddings

        elif isinstance(module, nn.Dropout):
            # Dropout layers do not need initialization
            pass

        # Recursively initialize weights for custom layers like ResBlock and SpatialTransformer
        elif hasattr(module, 'weight') and hasattr(module, 'bias'):
            if module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        #import pdb; pdb.set_trace()
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.temporal_encoder is not None:
            print(f"{self.__class__.__name__}: Also optimizing temporal encoder params!")
            params = params + list(self.temporal_encoder.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            # print('cattnCCC')
            # print(len(c_crossattn))
            # print(len(c_crossattn[0]))
            # print(len(c_crossattn[0][0]))
            if c_crossattn is not None:
                cc = torch.cat(c_crossattn, 1)
            else:
                cc = None
            # print('cattnaaaa', cc.shape)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            # print("hgybrid forward")
            # print(x.shape)
            # print(c_concat[0].shape)
            # print(c_concat.shape)
            xc = torch.cat([x, c_concat], dim=1)
            # print("HYBRID CALLED", xc.shape)
            # print(c_crossattn[0].shape)
            # print(c_crossattn.__class__)
            #cc = torch.tensor(c_crossattn) # torch.cat(c_crossattn, 1)
            
            cc = c_crossattn # torch.cat(c_crossattn, 1)
            # print("yeye ", cc.shape)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class Layout2ImgDiffusion(LatentDiffusion):
    # TODO: move all layout-specific hacks to this class
    def __init__(self, cond_stage_key, *args, **kwargs):
        assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

    def log_images(self, batch, N=8, *args, **kwargs):
        logs = super().log_images(batch=batch, N=N, *args, **kwargs)

        key = 'train' if self.training else 'validation'
        dset = self.trainer.datamodule.datasets[key]
        mapper = dset.conditional_builders[self.cond_stage_key]

        bbox_imgs = []
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        cond_img = torch.stack(bbox_imgs, dim=0)
        logs['bbox_image'] = cond_img
        return logs

def update_accuracy_csv(history_length, icon, accuracy, total_cases, csv_path='model_pred_icon_accuracy_vs_history.csv'):
    # Read existing CSV if it exists
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['history_length', 'icon', 'accuracy', 'total_cases'])
    
    # Check if entry exists
    mask = (df['history_length'] == history_length) & (df['icon'] == icon)
    
    if mask.any():
        # Update existing entry
        df.loc[mask, 'accuracy'] = accuracy
        df.loc[mask, 'total_cases'] = total_cases
    else:
        # Add new entry
        new_row = pd.DataFrame({
            'history_length': [history_length],
            'icon': [icon],
            'accuracy': [accuracy], 
            'total_cases': [total_cases]
        })
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Sort by history_length and icon
    df = df.sort_values(['history_length', 'icon']).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
