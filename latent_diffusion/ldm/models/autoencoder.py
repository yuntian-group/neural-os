import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from contextlib import contextmanager
from einops import rearrange
import numpy as np

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from latent_diffusion.ldm.modules.diffusionmodules.model import Encoder, Decoder
from latent_diffusion.ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from latent_diffusion.ldm.util import instantiate_from_config


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.automatic_optimization = False
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        opt_ae, opt_disc = self.optimizers()

        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), cond=None, split="train",
                                            )
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        opt_ae.zero_grad()  # Zero gradients for the autoencoder optimizer
        self.manual_backward(aeloss)  # Backpropagate the autoencoder loss
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) +
                      list(self.decoder.parameters()) +
                      list(self.quantize.parameters()) +
                      list(self.quant_conv.parameters()) +
                      list(self.post_quant_conv.parameters()),
                      1) #gradient clipping of 1

        opt_ae.step()  # Step the autoencoder optimizer

        # train the discriminator
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        opt_disc.zero_grad()  # Zero gradients for the discriminator optimizer
        self.manual_backward(discloss)  # Backpropagate the discriminator loss
        torch.nn.utils.clip_grad_norm_(self.loss.discriminator.parameters(), 1) #gradient clipping of 1
        opt_disc.step()  # Step the discriminator optimizer
        
        return {'aeloss': aeloss, 'discloss': discloss}

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # Update version check for lightning
        try:
            import pkg_resources
            version = pkg_resources.parse_version
            import lightning.pytorch as lightning_pkg
            if version(lightning_pkg.__version__) >= version('1.4.0'):
                del log_dict_ae[f"val{suffix}/rec_loss"]
        except (ImportError, AttributeError):
            pass  # Skip version check if it fails
        self.log_dict(log_dict_ae, sync_dist=True)
        self.log_dict(log_dict_disc, sync_dist=True)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        # Use automatic optimization when discriminator is disabled (for gradient accumulation support)
        # Use manual optimization when discriminator is enabled (need to handle two optimizers)
        self.automatic_optimization = not self.loss.use_discriminator
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        #self.load_state_dict(sd, strict=False)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, **kwargs):
        #import pdb; pdb.set_trace()
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        #import pdb; pdb.set_trace()

        # Manual checkpoint every 10 steps
        if self.global_step % 12000 == 0:
            print ('saving checkpoint')
            self.trainer.save_checkpoint(f"{self.trainer.save_dir}/model-{self.global_step:06d}.ckpt")
            print ('saved checkpoint to', f"{self.trainer.save_dir}/model-{self.global_step:06d}.ckpt")
        
        if self.loss.use_discriminator:
            # Manual optimization mode: handle both optimizers in a single step
            opt_ae, opt_disc = self.optimizers()
            
            # Step 1: train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            
            opt_ae.zero_grad()  # Zero gradients for the autoencoder optimizer
            self.manual_backward(aeloss)  # Backpropagate the autoencoder loss
            
            # Compute gradient norms before clipping
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), float('inf'))
            decoder_grad_norm = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), float('inf'))
            quant_conv_grad_norm = torch.nn.utils.clip_grad_norm_(self.quant_conv.parameters(), float('inf'))
            post_quant_conv_grad_norm = torch.nn.utils.clip_grad_norm_(self.post_quant_conv.parameters(), float('inf'))
            
            # Clip gradients
            total_ae_grad_norm = torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) +
                          list(self.decoder.parameters()) +
                          list(self.quant_conv.parameters()) +
                          list(self.post_quant_conv.parameters()),
                          1) #gradient clipping of 1
            opt_ae.step()  # Step the autoencoder optimizer
            
            # Step 2: train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            
            opt_disc.zero_grad()  # Zero gradients for the discriminator optimizer
            self.manual_backward(discloss)  # Backpropagate the discriminator loss
            
            # Compute discriminator gradient norm before clipping
            disc_grad_norm = torch.nn.utils.clip_grad_norm_(self.loss.discriminator.parameters(), float('inf'))
            
            # Clip discriminator gradients
            torch.nn.utils.clip_grad_norm_(self.loss.discriminator.parameters(), 1) #gradient clipping of 1
            opt_disc.step()  # Step the discriminator optimizer
            
            # Merge all metrics into a single log_dict call
            # Extract new-style loss (aeloss is already new-style from the loss function return)
            aeloss_new_style = aeloss  # This is already the new-style loss returned from loss function
            # Extract old-style loss for backward compatibility with existing plots
            aeloss_old_style = log_dict_ae.get("train/total_loss", torch.tensor(0.0))
            if isinstance(aeloss_old_style, torch.Tensor):
                aeloss_old_style = aeloss_old_style.item() if aeloss_old_style.numel() == 1 else aeloss_old_style
            
            log_dict = {
                "aeloss": aeloss_old_style,  # Old-style loss (for backward compatibility with existing plots)
                "aeloss_new_style": aeloss_new_style,  # New-style loss (properly averaged, used for training)
                "discloss": discloss,
                "grad_norm/encoder": encoder_grad_norm,
                "grad_norm/decoder": decoder_grad_norm,
                "grad_norm/quant_conv": quant_conv_grad_norm,
                "grad_norm/post_quant_conv": post_quant_conv_grad_norm,
                "grad_norm/total_ae": total_ae_grad_norm,
                "grad_norm/discriminator": disc_grad_norm,
                **log_dict_ae,
                **log_dict_disc
            }
            self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            
            return {'aeloss': aeloss, 'discloss': discloss}
        else:
            # Automatic optimization mode: single optimizer, Lightning handles accumulation
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            
            # DEBUG: Log raw rec_loss to verify it's being computed correctly
            raw_rec_loss = torch.abs(inputs - reconstructions).mean()
            if self.global_step % 100 == 0:
                print(f"DEBUG step {self.global_step}: raw_rec_loss={raw_rec_loss.item():.6f}, aeloss={aeloss.item():.6f}, kl_loss={log_dict_ae.get('train/kl_loss_new_style', torch.tensor(0.0)).item():.6f}")
            
            # Skip discriminator training when disabled (for efficiency)
            discloss = torch.tensor(0.0, device=inputs.device)
            log_dict_disc = {
                "train/disc_loss": torch.tensor(0.0, device=inputs.device),
                "train/logits_real": torch.tensor(0.0, device=inputs.device),
                "train/logits_fake": torch.tensor(0.0, device=inputs.device)
            }
            
            # Store for logging in on_before_optimizer_step hook
            self._current_log_dict_ae = log_dict_ae
            self._current_log_dict_disc = log_dict_disc
            self._current_aeloss = aeloss
            self._current_discloss = discloss
            
            # Return loss for automatic optimization (Lightning handles backward/step)
            return aeloss
    
    def on_before_optimizer_step(self, optimizer):
        """Hook called before optimizer step - used for gradient clipping and logging in automatic mode"""
        if not self.loss.use_discriminator:
            # DEBUG: Check learning rate to see if checkpoint loading overwrote it
            if self.global_step % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                expected_lr = self.learning_rate
                print(f"DEBUG on_before_optimizer_step step {self.global_step}: optimizer.lr={current_lr:.2e}, self.learning_rate={expected_lr:.2e}, match={abs(current_lr - expected_lr) < 1e-10}")
                import pdb; pdb.set_trace()  # Breakpoint to inspect
            
            # Automatic optimization mode: compute gradient norms and clip
            # Compute gradient norms before clipping
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), float('inf'))
            decoder_grad_norm = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), float('inf'))
            quant_conv_grad_norm = torch.nn.utils.clip_grad_norm_(self.quant_conv.parameters(), float('inf'))
            post_quant_conv_grad_norm = torch.nn.utils.clip_grad_norm_(self.post_quant_conv.parameters(), float('inf'))
            
            # Clip gradients
            total_ae_grad_norm = torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) +
                          list(self.decoder.parameters()) +
                          list(self.quant_conv.parameters()) +
                          list(self.post_quant_conv.parameters()),
                          1) #gradient clipping of 1
            
            disc_grad_norm = torch.tensor(0.0, device=self.device)
            
            # Extract losses for logging
            aeloss_new_style = self._current_aeloss
            aeloss_old_style = self._current_log_dict_ae.get("train/total_loss", torch.tensor(0.0))
            if isinstance(aeloss_old_style, torch.Tensor):
                aeloss_old_style = aeloss_old_style.item() if aeloss_old_style.numel() == 1 else aeloss_old_style
            
            # Log metrics
            log_dict = {
                "aeloss": aeloss_old_style,
                "aeloss_new_style": aeloss_new_style,
                "discloss": self._current_discloss,
                "grad_norm/encoder": encoder_grad_norm,
                "grad_norm/decoder": decoder_grad_norm,
                "grad_norm/quant_conv": quant_conv_grad_norm,
                "grad_norm/post_quant_conv": post_quant_conv_grad_norm,
                "grad_norm/total_ae": total_ae_grad_norm,
                "grad_norm/discriminator": disc_grad_norm,
                **self._current_log_dict_ae,
                **self._current_log_dict_disc
            }
            self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        if self.loss.use_discriminator:
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")
        else:
            discloss = torch.tensor(0.0, device=inputs.device)
            log_dict_disc = {
                "val/disc_loss": torch.tensor(0.0, device=inputs.device),
                "val/logits_real": torch.tensor(0.0, device=inputs.device),
                "val/logits_fake": torch.tensor(0.0, device=inputs.device)
            }

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], sync_dist=True)
        self.log_dict(log_dict_ae, sync_dist=True)
        self.log_dict(log_dict_disc, sync_dist=True)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        print ('learning rate ----', lr)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        
        if self.loss.use_discriminator:
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
            return [opt_ae, opt_disc], []
        else:
            return opt_ae
    
    def on_load_checkpoint(self, checkpoint):
        """Override learning rate from checkpoint with config value"""
        # Update optimizer learning rates to match config (checkpoint may have old LR)
        if 'optimizer_states' in checkpoint:
            expected_lr = self.learning_rate
            for opt_state in checkpoint['optimizer_states']:
                if 'param_groups' in opt_state:
                    for param_group in opt_state['param_groups']:
                        old_lr = param_group.get('lr', None)
                        if old_lr is not None and abs(old_lr - expected_lr) > 1e-10:
                            print(f"Overriding checkpoint LR {old_lr:.2e} with config LR {expected_lr:.2e}")
                            param_group['lr'] = expected_lr
    
    def on_train_start(self):
        """Ensure optimizer learning rate matches config after checkpoint loading"""
        expected_lr = self.learning_rate
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizers = optimizer
        else:
            optimizers = [optimizer]
        
        for opt in optimizers:
            for param_group in opt.param_groups:
                current_lr = param_group['lr']
                if abs(current_lr - expected_lr) > 1e-10:
                    print(f"Fixing optimizer LR from {current_lr:.2e} to {expected_lr:.2e}")
                    param_group['lr'] = expected_lr

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

