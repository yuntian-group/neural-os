from latent_diffusion.ldm.models.autoencoder import VQModel

from data.data_processing.datasets import DataModule
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from omegaconf import OmegaConf
from typing import List
import argparse, os
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

#from computer.util import LoggingCallback

def train_model(model: VQModel, data: DataModule, save_path: str, config: OmegaConf, ckpt_path) -> VQModel:

    """
    Trains the decoder layers in the autoencoder on specified dataset in the config.
    """

    os.makedirs(save_path, exist_ok=True)

    # Define a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=1000,
        dirpath=save_path,             # Directory to save checkpoints
        filename='model-{step:06d}',  # Checkpoint filename format
        save_top_k=-1,                 # Save all checkpoints, not just the best one
        #save_weights_only=False,       # Save full model (weights + optimizer state)
    )

    # If resuming from checkpoint, set the starting step for the callback
    ckpt_has_optimizer_states = False
    if ckpt_path:
        torch.serialization.add_safe_globals([ModelCheckpoint])
        ckpt = torch.load(ckpt_path, map_location='cpu')
        global_step = ckpt.get('global_step', 0)
        checkpoint_callback.last_model_path = os.path.join(save_path, f'model-{global_step:06d}.ckpt')
        checkpoint_callback.current_score = None
        checkpoint_callback.best_k_models = {}
        checkpoint_callback.kth_best_model_path = ''
        checkpoint_callback.best_model_score = None
        checkpoint_callback.best_model_path = ''
        
        # Check if checkpoint has optimizer states
        ckpt_has_optimizer_states = 'optimizer_states' in ckpt
        if ckpt_has_optimizer_states:
            print(f"Resuming from step {global_step} (with optimizer states)")
        else:
            print(f"Loading model weights from step {global_step} (no optimizer states, starting fresh)")
            # Load model weights manually if no optimizer states
            # Lightning checkpoints store model state_dict under 'state_dict' key
            if 'state_dict' in ckpt:
                # Extract model state_dict (remove 'model.' prefix if present)
                state_dict = ckpt['state_dict']
                # Handle nested state_dict structure
                if any(k.startswith('model.') for k in state_dict.keys()):
                    # Remove 'model.' prefix from keys
                    state_dict = {k.replace('model.', '', 1) if k.startswith('model.') else k: v 
                                 for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
            else:
                # Fallback: try loading entire checkpoint as state_dict
                model.load_state_dict(ckpt, strict=False)
            print("Model weights loaded successfully")
        del ckpt  # Free memory

    #Disables training for the encoder side.
    #model.encoder.eval()
    #model.encoder.train = disabled_train
    #for param in model.encoder.parameters():
    #    param.requires_grad = False

    # configure learning rate
    base_lr = config.model.base_learning_rate
    model.learning_rate = base_lr

    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    trainer_opt = argparse.Namespace(**trainer_config)

    print(trainer_opt)

    # Remove checkpoint_callback from trainer_opt if it exists
    # to avoid "unexpected keyword argument" error
    trainer_kwargs = vars(trainer_opt)
    if 'checkpoint_callback' in trainer_kwargs:
        del trainer_kwargs['checkpoint_callback']

    trainer: Trainer = Trainer(**trainer_kwargs, callbacks=[checkpoint_callback])
    trainer.save_dir = save_path

    print("\u2705 Fitting model...")

    if ckpt_path and ckpt_has_optimizer_states:
        print('Loading checkpoint with optimizer states')
        # Load checkpoint with optimizer states
        trainer.fit(model, data, ckpt_path=ckpt_path)
    elif ckpt_path and not ckpt_has_optimizer_states:
        print('Loading checkpoint without optimizer states')
        # Model weights already loaded manually, start training fresh
        trainer.fit(model, data)
    else:
        print('No checkpoint, starting from scratch')
        # No checkpoint, start from scratch
        trainer.fit(model, data)
    
    if trainer.is_global_zero: print(f"\u2705 Checkpoints saved at {save_path}")
    
    return trainer.model, trainer.is_global_zero
    
