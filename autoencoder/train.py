from latent_diffusion.ldm.models.autoencoder import VQModel

from data.data_processing.datasets import DataModule
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from omegaconf import OmegaConf
from typing import List
import argparse, os
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

#from computer.util import LoggingCallback

def train_model(model: VQModel, data: DataModule, save_path: str, config: OmegaConf, ckpt_path) -> VQModel:

    """
    Trains the decoder layers in the autoencoder on specified dataset in the config.
    """

    os.makedirs(save_path, exist_ok=True)

    # Define a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=10,
        dirpath=save_path,             # Directory to save checkpoints
        filename='model-{step:06d}',  # Checkpoint filename format
        save_top_k=-1,                 # Save all checkpoints, not just the best one
        #save_weights_only=False,       # Save full model (weights + optimizer state)
    )

    # If resuming from checkpoint, set the starting step for the callback
    if ckpt_path:
        torch.serialization.add_safe_globals([ModelCheckpoint])
        ckpt = torch.load(ckpt_path)
        global_step = ckpt['global_step']
        checkpoint_callback.last_model_path = os.path.join(save_path, f'model-{global_step:06d}.ckpt')
        checkpoint_callback.current_score = None
        checkpoint_callback.best_k_models = {}
        checkpoint_callback.kth_best_model_path = ''
        checkpoint_callback.best_model_score = None
        checkpoint_callback.best_model_path = ''
        print(f"Resuming from step {global_step}")

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

    trainer: Trainer = Trainer(**vars(trainer_opt), callbacks=[checkpoint_callback])
    trainer.save_dir = save_path

    print("\u2705 Fitting model...")

    if ckpt_path:
        trainer.fit(model, data, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, data)
    
    if trainer.is_global_zero: print(f"\u2705 Checkpoints saved at {save_path}")
    
    return trainer.model, trainer.is_global_zero
    
