import pandas as pd               # For reading CSV files with image and MRI paths.
import numpy as np                # For handling numerical operations, arrays, and data types.
import torchvision.transforms as transforms  # For performing image augmentations like random horizontal flips.
from torch.utils.data import Dataset  # Dataset class to inherit for custom datasets.
import os
import torch
from typing import List
from latent_diffusion.ldm.modules.encoders.modules import BERTTokenizer
import ast
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from PIL import Image

from latent_diffusion.ldm.util import instantiate_from_config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def parse_action_string(action_str):
    """Convert formatted action string to x, y coordinates
    Args:
        action_str: String like 'N N N N N : N N N N N' or '+ 0 2 1 3 : + 0 3 8 3'
    Returns:
        tuple: (x, y) coordinates or None if action is padding
    """
    if 'N' in action_str:
        return (None, None)
        
    # Split into x and y parts
    action_str = action_str.replace(' ', '')
    x_part, y_part = action_str.split(':')
    
    # Parse x: remove sign, join digits, convert to int, apply sign
    
    x = int(x_part)
    
    # Parse y: remove sign, join digits, convert to int, apply sign
    y = int(y_part)
    
    return (x, y)

def create_position_map(pos, image_size=64, original_width=1024, original_height=640):
    """Convert cursor position to a binary position map
    Args:
        x, y: Original cursor positions
        image_size: Size of the output position map (square)
        original_width: Original screen width (1024)
        original_height: Original screen height (640)
    Returns:
        torch.Tensor: Binary position map of shape (1, image_size, image_size)
    """
    x, y = pos
    if x is None:
        return torch.zeros((1, image_size, image_size))
    # Scale the positions to new size
    x_scaled = int((x / original_width) * image_size)
    y_scaled = int((y / original_height) * image_size)
    
    # Clamp values to ensure they're within bounds
    x_scaled = max(0, min(x_scaled, image_size - 1))
    y_scaled = max(0, min(y_scaled, image_size - 1))
    
    # Create binary position map
    pos_map = torch.zeros((1, image_size, image_size))
    #pos_map[0, y_scaled, x_scaled] = 1.0
    pos_map[0, x_scaled, y_scaled] = 1.0
    
    
    return pos_map

class ActionsData(Dataset):
    """
    class dataset for csllm. includes image sequences and corresponding action sequences for cond.
    """
    def __init__(self,
                 data_csv_path,
                 ):
        self.data_path = data_csv_path
        
        data = pd.read_csv(data_csv_path)
        self.image_seq_paths = data["Image_seq_cond_path"].apply(ast.literal_eval).to_list()
        self.actions_seq = data['Action_seq'].apply(ast.literal_eval).to_list()
        self.targets = data['Target_image'].to_list()

        self._length = len(self.image_seq_paths)


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        """
        takes a sequence of cond. images and actions and a single target.
        """
        example = dict()
        i = i % self._length
        # i = 0 if i % 2 == 0 else 50


        #single sample overfit
        example["image"] = normalize_image(self.targets[i]) # torch.stack(image_target) # n b w h c
        action_seq = self.actions_seq[i]
        assert len(action_seq) == 15, "Action sequence must be 15 actions long"
        for j in range(8):
            example[f"action_{j}"] = action_seq[j:j+8]
            assert len(example[f"action_{j}"]) == 8, f"Action sequence {j} must be 8 actions long"
            example[f"action_{j}"] = ' '.join(example[f"action_{j}"])
            example[f"position_map_{j}"] = create_position_map(parse_action_string(action_seq[j+7]))
        #example["caption"] = ' '.join(self.actions_seq[i]) # actions_cond #untokenized actions

        example['c_concat'] = torch.stack([normalize_image(image_path) for image_path in self.image_seq_paths[i]]) # sequence of images

        return example 

def normalize_image(image_path : str | Image.Image): 

    """
    Takes in an image path or an image and returns the normalized image in a tensor.
    """
    
    if isinstance(image_path, str): image = Image.open(image_path)
    else: image = image_path

    if not image.mode == "RGB":
        image = image.convert("RGB")

    image = (np.array(image) / 127.5 - 1.0).astype(np.float32)

    return torch.tensor(image)


class DataModule(pl.LightningDataModule):

    """
    This is the module we pass to the trainer. 
    """

    def __init__(self, 
                 batch_size, 
                 train=None, 
                 validation=None, 
                 test=None, 
                 wrap=False, 
                 num_workers=None, 
                 shuffle=True,
                 drop_last=False,
                 pin_memory=False,
                 prefetch_factor=2,
                 persistent_workers=False
        ):
        super().__init__()
        self.batch_size = batch_size
        self.wrap=wrap, 
        self.num_workers=num_workers
        self.shuffle=shuffle,
        self.drop_last=drop_last,
        self.pin_memory=pin_memory,
        self.prefetch_factor=prefetch_factor,
        self.persistent_workers=persistent_workers

        if num_workers > 1:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"

        self.dataset_configs = dict()

        if train:
            self.dataset_configs["train"] = train
            self.train_dataloader = self.init_train_dataloader
        if validation:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self.init_val_dataloader
        if test:
            self.dataset_configs["test"] = test
            self.test_dataloader = self.init_test_dataloader

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def init_train_dataloader(self):
        return DataLoader(self.datasets["train"], 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, 
                          shuffle=self.shuffle,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          drop_last=True)

    def init_val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=self.shuffle,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          drop_last=True)

    def init_test_dataloader(self):
        return DataLoader(self.datasets["test"], 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, 
                          shuffle=self.shuffle,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          drop_last=True)
        
