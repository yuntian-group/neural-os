from random import shuffle
from data.data_processing.datasets import ActionsSequenceData
from latent_diffusion.ldm.models.diffusion.ddpm import DDIMSampler, LatentDiffusion, disabled_train
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from typing import List
import argparse, os
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
    
def train_model(model: LatentDiffusion, data: ActionsSequenceData, save_path: str, config) -> LatentDiffusion:

    """
    Trains the model on specified dataset in the config.
    """

     # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate

    model.learning_rate = base_lr

    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    trainer_opt = argparse.Namespace(**trainer_config)

    trainer: Trainer= Trainer.from_argparse_args(trainer_opt)

    data_l = DataLoader(data, batch_size=bs, num_workers=4, shuffle=False) #TODO Pass this to the trainer

    if data_l.num_workers > 1: os.environ["TOKENIZERS_PARALLELISM"] = "false"

    trainer.fit(model, data_l)
    os.makedirs(save_path, exist_ok=True)
    trainer.save_checkpoint(f"{save_path}/model_{save_path}.ckpt")
    print(f"Saved {save_path}/model_{save_path}.ckpt")

    return trainer.model

