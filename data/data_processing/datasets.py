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
from einops import rearrange
from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config

from torch.utils.data import DataLoader, Dataset
from PIL import Image

from latent_diffusion.ldm.util import instantiate_from_config
#from computer.util import load_model_from_config

import cv2
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from cairosvg import svg2png  # You might need to pip install cairosvg

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def parse_action_string(action_str):
    """Convert formatted action string to x, y coordinates
    Args:
        action_str: String like 'N N N N N : N N N N N' or '+ 0 2 1 3 : + 0 3 8 3'
    Returns:
        tuple: (x, y) coordinates or None if action is padding
    """
    action_type = action_str[0]
    action_str = action_str[1:].strip()
    if 'N' in action_str:
        return (None, None, None)
        
    # Split into x and y parts
    action_str = action_str.replace(' ', '')
    x_part, y_part = action_str.split(':')
    
    # Parse x: remove sign, join digits, convert to int, apply sign
    
    x = int(x_part)
    
    # Parse y: remove sign, join digits, convert to int, apply sign
    y = int(y_part)
    
    return x, y, action_type

def create_position_and_click_map(pos, action_type, image_height=48, image_width=64, original_width=512, original_height=384):
    """Convert cursor position to a binary position map
    Args:
        x, y: Original cursor positions
        image_width: Width of the output position map (64)
        image_height: Height of the output position map (48)
        original_width: Original screen width (1024)
        original_height: Original screen height (640)
    Returns:
        torch.Tensor: Binary position map of shape (1, image_height, image_width)
    """
    x, y = pos
    if x is None:
        return torch.zeros((1, image_height, image_width)), torch.zeros((1, image_height, image_width))
    
    x_scaled = int(x / original_width * image_width)
    y_scaled = int(y / original_height * image_height)
    
    # Clamp values to ensure they're within bounds
    x_scaled = max(0, min(x_scaled, image_width - 1))
    y_scaled = max(0, min(y_scaled, image_height - 1))
    
    # Create binary position map
    pos_map = torch.zeros((1, image_height, image_width))
    pos_map[0, y_scaled, x_scaled] = 1.0

    leftclick_map = torch.zeros((1, image_height, image_width))
    if action_type == 'L':
        leftclick_map[0, y_scaled, x_scaled] = 1.0
    
    return pos_map, leftclick_map

def get_cursor_image():
    """Get cursor image from SVG"""
    cursor_svg = '''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
     viewBox="8 4 12 20" enable-background="new 8 4 12 20" xml:space="preserve">
<polygon fill="#FFFFFF" points="8.2,20.9 8.2,4.9 19.8,16.5 13,16.5 12.6,16.6 "/>
<polygon fill="#FFFFFF" points="17.3,21.6 13.7,23.1 9,12 12.7,10.5 "/>
<rect x="12.5" y="13.6" transform="matrix(0.9221 -0.3871 0.3871 0.9221 -5.7605 6.5909)" width="2" height="8"/>
<polygon points="9.2,7.3 9.2,18.5 12.2,15.6 12.6,15.5 17.4,15.5 "/>
</svg>'''
    
    png_data = svg2png(bytestring=cursor_svg.encode('utf-8'), 
                      output_width=12, 
                      output_height=20)
    
    cursor = Image.open(BytesIO(png_data))
    cursor_array = np.array(cursor)
    
    cursor_with_outline = Image.new('RGBA', cursor.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(cursor_with_outline)
    
    black_mask = np.all(cursor_array == [255, 255, 255, 255], axis=-1)
    outline_positions = np.where(black_mask)
    
    for y, x in zip(*outline_positions):
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < cursor.width and 0 <= new_y < cursor.height:
                if not black_mask[new_y, new_x]:
                    cursor_with_outline.putpixel((new_x, new_y), (0, 0, 0, 255))
    
    cursor_with_outline.alpha_composite(cursor)
    return np.array(cursor_with_outline)

def draw_cursor(frame, x, y, left_click=False, right_click=False, scaling_factor=1):
    """Draw a cursor on the frame at the given position"""
    x, y = int(x * scaling_factor), int(y * scaling_factor)
    
    cursor = get_cursor_image()
    h, w = cursor.shape[:2]
    
    frame_h, frame_w = frame.shape[:2]
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(frame_w, x + w)
    y_end = min(frame_h, y + h)
    
    cursor_x_start = max(0, -x)
    cursor_y_start = max(0, -y)
    cursor_x_end = cursor_x_start + (x_end - x_start)
    cursor_y_end = cursor_y_start + (y_end - y_start)
    
    if x_end > x_start and y_end > y_start:
        alpha = cursor[cursor_y_start:cursor_y_end, cursor_x_start:cursor_x_end, 3] / 255.0
        alpha = alpha[..., np.newaxis]
        
        cursor_part = cursor[cursor_y_start:cursor_y_end, cursor_x_start:cursor_x_end, :3]
        frame_part = frame[y_start:y_end, x_start:x_end]
        blended = (cursor_part * alpha + frame_part * (1 - alpha)).astype(np.uint8)
        frame[y_start:y_end, x_start:x_end] = blended
        
        if left_click or right_click:
            click_radius = 4
            click_color = (0, 255, 255) if left_click else (255, 255, 0)
            cv2.circle(frame, (x + 8, y + 8), click_radius, click_color, -1)
    
    return frame
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
class ActionsData(Dataset):
    """
    class dataset for csllm. includes image sequences and corresponding action sequences for cond.
    """
    def __init__(self,
                 data_csv_path,
                 debug_mode=False
                 ):
        self.data_path = data_csv_path
        self.debug_mode = debug_mode
        
        # Read only the necessary columns
        data = pd.read_csv(data_csv_path, usecols=["Image_seq_cond_path", "Action_seq", "Target_image"])
        self.image_seq_paths = data["Image_seq_cond_path"].apply(ast.literal_eval).to_list()
        self.actions_seq = data['Action_seq'].apply(ast.literal_eval).to_list()
        self.targets = data['Target_image'].to_list()
        
        # Clear the dataframe from memory
        del data
        
        self._length = len(self.image_seq_paths)
        
        # Check if processed data exists by checking first image
        first_img = self.image_seq_paths[0][0]
        processed_path = first_img.replace('train_dataset/', 'train_dataset_encoded/').replace('.png', '.npy')
        self.use_processed = os.path.exists(processed_path)
        if self.use_processed:
            print("Found processed data in train_dataset_encoded/")
            
            # Load model for reprocessing if needed
            #try:
            if 'train' in data_csv_path:
                print ('Loading autoencoder model for reprocessing')
                config = OmegaConf.load("autoencoder_config_kl4_lr4.5e6_load_acc1.yaml")
                self.model = load_model_from_config(config, "autoencoder_saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_model-603000.ckpt")
                #self.model.load_state_dict(torch.load("autoencoder_saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_model-603000.ckpt"))
                #self.model = self.model.to(device)
                self.model.eval()
                print("Loaded model for reprocessing if needed")
            else:
                self.model = None
            #except Exception as e:
            #    print(f"Warning: Could not load model for reprocessing: {e}")
            #    self.model = None

    def __len__(self):
        return self._length

    def load_processed_image(self, image_path):
        """Load preprocessed latent from .npy file, reprocess if loading fails"""
        processed_path = image_path.replace('train_dataset/', 'train_dataset_encoded/').replace('.png', '.npy')
        try:
            return torch.from_numpy(np.load(processed_path))
        except Exception as e:
            print(f"Warning: Failed to load {processed_path}, reprocessing...")
            print(e)
            
            if self.model is None:
                raise RuntimeError("Model not available for reprocessing")
            
            # Load and process the original image
            image = normalize_image(image_path)
            image = torch.unsqueeze(image, dim=0)
            image = rearrange(image, 'b h w c -> b c h w')#.to(device)
            
            # Get latent representation
            with torch.no_grad():
                posterior = self.model.encode(image)
                latent = posterior.sample()
                
                # Special handling for padding.png
                if os.path.basename(image_path) == 'padding.png':
                    latent = torch.zeros_like(latent)
                latent = latent.squeeze(0)
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(processed_path), exist_ok=True)
                # Save the processed latent
                np.save(processed_path, latent.cpu().numpy())
            except Exception as e:
                print(f"Warning: Failed to save {processed_path}")
                print(e)
                
            return latent

    def __getitem__(self, i):
        """
        takes a sequence of cond. images and actions and a single target.
        Always loads original images, and loads processed versions if available
        """
        #print ('getitem', i)
        example = dict()
        i = i % self._length
        
        if self.debug_mode:
            assert False
            # Create a blank 64x64 image
            image = np.ones((64, 64, 3), dtype=np.uint8) * 255
            
            # Get the last action (current position)
            action = self.actions_seq[i][-1]
            coords = parse_action_string(action)
            
            if coords is not None:
                x, y = coords
                # Scale coordinates from 1024x640 to 64x64
                x_scaled = int((x / 1024) * 64)
                y_scaled = int((y / 640) * 64)
                # Draw cursor at scaled position
                image = draw_cursor(image, x_scaled, y_scaled, scaling_factor=1)
            
            example["image"] = normalize_image(Image.fromarray(image))
        else:
            # Always load original images
            
            
            # Load processed versions if available
            if self.use_processed:
                example['image_processed'] = self.load_processed_image(self.targets[i])
                example['c_concat_processed'] = torch.stack([
                    self.load_processed_image(image_path) 
                    for image_path in self.image_seq_paths[i]
                ])
            else:
                example["image"] = normalize_image(self.targets[i])
                example['c_concat'] = torch.stack([normalize_image(image_path) 
                                                for image_path in self.image_seq_paths[i]])
        # Rest of the original code...
        action_seq = self.actions_seq[i]
        if len(action_seq) > 15:
            action_seq = action_seq[-15:]
        assert len(action_seq) == 15, "Action sequence must be 15 actions long"
        for j in range(8):
            example[f"action_{j}"] = action_seq[j:j+8]
            assert len(example[f"action_{j}"]) == 8, f"Action sequence {j} must be 8 actions long"
            example[f"action_{j}"] = ' '.join(example[f"action_{j}"])
            x, y, action_type = parse_action_string(action_seq[j+7])
            position_map, leftclick_map = create_position_and_click_map((x,y), action_type)
            example[f"position_map_{j}"] = position_map
            example[f"leftclick_map_{j}"] = leftclick_map
        for j in range(-1, -8, -1):
            x, y, action_type = parse_action_string(action_seq[j+7])
            position_map, leftclick_map = create_position_and_click_map((x,y), action_type)
            example[f"position_map_{j}"] = position_map
            example[f"leftclick_map_{j}"] = leftclick_map

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
        
