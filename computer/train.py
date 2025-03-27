from random import shuffle
from data.data_processing.datasets import ActionsData, DataModule
from latent_diffusion.ldm.models.diffusion.ddpm import DDIMSampler, LatentDiffusion, disabled_train
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from typing import List
import argparse, os
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

def train_model(model: LatentDiffusion, data: DataModule, save_path: str, config) -> LatentDiffusion:
    """
    Trains the model on specified dataset in the config.
    """

    # configure learning rate
    base_lr = config.model.base_learning_rate
    model.learning_rate = base_lr

    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_opt = argparse.Namespace(**trainer_config)

    # Create ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=2000, # orig 1000
        save_top_k=-1,  # Save all checkpoints
        dirpath=save_path,  # Directory to save checkpoints
        filename='model-{step:06d}'  # Checkpoint filename format
    )

    # Create Trainer with the checkpoint callback
    trainer: Trainer = Trainer(**vars(trainer_opt), callbacks=[checkpoint_callback])

    trainer.fit(model, data)
   
    if 'eval' not in save_path:
        os.makedirs(save_path, exist_ok=True)
        trainer.save_checkpoint(f"{save_path}/model_{save_path}.ckpt")
        print(f"Saved {save_path}/model_{save_path}.ckpt")

    return trainer.model
    
