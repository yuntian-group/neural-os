import pandas as pd               # For reading CSV files with image and MRI paths.
import numpy as np                # For handling numerical operations, arrays, and data types.
import torchvision.transforms as transforms  # For performing image augmentations like random horizontal flips.
from torch.utils.data import Dataset  # Dataset class to inherit for custom datasets.
import os
import torch
from typing import List
from latent_diffusion.ldm.modules.encoders.modules import BERTTokenizer
import ast
import lightning.pytorch as pl
from einops import rearrange
from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config
import json

from torch.utils.data import DataLoader, Dataset
from PIL import Image

from latent_diffusion.ldm.util import instantiate_from_config
#from computer.util import load_model_from_config

import cv2
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from cairosvg import svg2png  # You might need to pip install cairosvg
import pickle
import webdataset as wds
import functools
from collections import OrderedDict

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
    _shared_data = {}  # Class-level cache for dataframes
    
    def __init__(self,
                 data_csv_paths,
                 use_original_image=False,
                 debug_mode=False,
                 normalization='none',  # Options: 'none', 'minmax', 'standard'
                 context_length=7,  # New parameter for flexible context length
                 latent_stats_path='../latent_stats.json'  # Path to the JSON file with per-channel stats
                 ):
        self.data_paths = data_csv_paths
        self.debug_mode = debug_mode
        self._length = None  # Will be set in setup
        self.use_processed = False  # Will be set in setup
        self.normalization = normalization
        self.context_length = context_length
        self.use_original_image = use_original_image
        self.latent_stats_path = latent_stats_path

        # Load action mapping
        self.mapping_dicts = []
        self.base_dirs = []
        for (i, data_csv_path) in enumerate(self.data_paths):
            base_dir = os.path.dirname(data_csv_path)
            mapping_dict_path = os.path.join(base_dir, f'image_action_mapping_with_key_states.pkl')
            with open(mapping_dict_path, 'rb') as f:
                self.mapping_dicts.append(pickle.load(f))
            self.base_dirs.append(base_dir)
        
        #with open('../computer/image_action_mapping_with_key_states.pkl', 'rb') as f:
        #    self.mapping_dict = pickle.load(f)
        
        # Constants for normalization (based on your analysis)
        self.data_mean = -0.54
        self.data_std = 6.78
        self.data_min = -27.681446075439453
        self.data_max = 30.854148864746094
        
        # Load per-channel statistics if available
        self.per_channel_mean = None
        self.per_channel_std = None
        if latent_stats_path and os.path.exists(latent_stats_path):
            with open(latent_stats_path, 'r') as f:
                stats = json.load(f)
            self.per_channel_mean = torch.tensor(stats['mean'])
            self.per_channel_std = torch.tensor(stats['std'])
            print(f"Loaded per-channel statistics from {latent_stats_path}")
            print(f"Mean shape: {self.per_channel_mean.shape}, Std shape: {self.per_channel_std.shape}")
           

        KEYS = ['\t', '\n', '\r', ' ', '!', '"', '#', '$', '%', '&', "'", '(',
        ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~',
        'accept', 'add', 'alt', 'altleft', 'altright', 'apps', 'backspace',
        'browserback', 'browserfavorites', 'browserforward', 'browserhome',
        'browserrefresh', 'browsersearch', 'browserstop', 'capslock', 'clear',
        'convert', 'ctrl', 'ctrlleft', 'ctrlright', 'decimal', 'del', 'delete',
        'divide', 'down', 'end', 'enter', 'esc', 'escape', 'execute', 'f1', 'f10',
        'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f2', 'f20',
        'f21', 'f22', 'f23', 'f24', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
        'final', 'fn', 'hanguel', 'hangul', 'hanja', 'help', 'home', 'insert', 'junja',
        'kana', 'kanji', 'launchapp1', 'launchapp2', 'launchmail',
        'launchmediaselect', 'left', 'modechange', 'multiply', 'nexttrack',
        'nonconvert', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6',
        'num7', 'num8', 'num9', 'numlock', 'pagedown', 'pageup', 'pause', 'pgdn',
        'pgup', 'playpause', 'prevtrack', 'print', 'printscreen', 'prntscrn',
        'prtsc', 'prtscr', 'return', 'right', 'scrolllock', 'select', 'separator',
        'shift', 'shiftleft', 'shiftright', 'sleep', 'space', 'stop', 'subtract', 'tab',
        'up', 'volumedown', 'volumemute', 'volumeup', 'win', 'winleft', 'winright', 'yen',
        'command', 'option', 'optionleft', 'optionright']
        INVALID_KEYS = ['f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'select', 'separator', 'execute']
        KEYS = [key for key in KEYS if key not in INVALID_KEYS]

        self.itos = KEYS
        self.stoi = {key: i for i, key in enumerate(KEYS)}

        
    def setup(self):
        print ('setup called')
        for (i, data_path) in enumerate(self.data_paths):
            if data_path not in ActionsData._shared_data:
                print(f"Loading data from {data_path}")
                # Read target frames data
                data = pd.read_csv(data_path)
                ActionsData._shared_data[data_path] = {
                    'record_nums': data['record_num'].tolist(),
                    'image_nums': data['image_num'].tolist()
                }
                del data
            
        # Get data from cache
        cached_datas = []
        for (i, data_path) in enumerate(self.data_paths):
            cached_data = ActionsData._shared_data[data_path]
            cached_datas.append(cached_data)
        self.record_nums_list = []
        for (i, data_path) in enumerate(self.data_paths):
            self.record_nums_list.append(cached_datas[i]['record_nums'])
        self.image_nums_list = []
        for (i, data_path) in enumerate(self.data_paths):
            self.image_nums_list.append(cached_datas[i]['image_nums'])
        self._lengths = []
        for (i, data_path) in enumerate(self.data_paths):
            self._lengths.append(len(self.record_nums_list[i]))
        
        # Always use processed data with WebDataset
        self.use_processed = True
        
        # Load model for reprocessing if needed
        if False and 'train' in self.data_path:
            print('Loading autoencoder model for reprocessing')
            config = OmegaConf.load("../autoencoder/config_kl4_lr4.5e6_load_acc1_512_384.yaml")
            #self.model = load_model_from_config(config, "autoencoder_saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_model-603000.ckpt")
            self.model = load_model_from_config(config, "../autoencoder/saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384/model-354000.ckpt")
            #self.model.load_state_dict(torch.load("autoencoder_saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_model-603000.ckpt"))
            #self.model = self.model.to(device)
            self.model.eval()
            print("Loaded model for reprocessing if needed")
        else:
            self.model = None
        #except Exception as e:
        #    print(f"Warning: Could not load model for reprocessing: {e}")
        #    self.model = None

    def normalize_features(self, x):
        """Normalize features based on specified strategy"""
        if self.normalization == 'none':
            return x
        elif self.normalization == 'minmax':
            # Normalize to [-1, 1]
            return 2.0 * (x - self.data_min) / (self.data_max - self.data_min) - 1.0
        elif self.normalization == 'standard' or 'standard' in self.normalization:
            # Check if we have per-channel statistics
            assert len(x.shape) == 4 or len(x.shape) == 3
            if self.per_channel_mean is not None and self.per_channel_std is not None:
                # Add singleton dimensions to match the latent representation
                if len(x.shape) == 4:
                    # For batched latent vectors (multiple images)
                    # Reshape mean and std to match the channel dimension
                    mean = self.per_channel_mean.view(1, -1, 1, 1)
                    std = self.per_channel_std.view(1, -1, 1, 1)
                elif len(x.shape) == 3:
                    # For a single latent vector
                    mean = self.per_channel_mean.view(-1, 1, 1)
                    std = self.per_channel_std.view(-1, 1, 1)
                # Apply per-channel normalization
                return (x - mean) / std
            else:
                assert False
                # Fall back to global normalization
                return (x - self.data_mean) / self.data_std
        else:
            raise ValueError(f"Unknown normalization strategy: {self.normalization}")

    def __len__(self):
        if self.debug_mode:
            assert False
            return self._length
        else:
            return sum(self._lengths)

    def __getitem__(self, i):
        example = dict()
        # Local cache for tar files used in this call
        local_tar_cache = {}
        
        i = i % sum(self._lengths)
        for (data_partition, length) in enumerate(self._lengths):
            if i < length:
                break
            i -= length
        record_nums = self.record_nums_list[data_partition]
        image_nums = self.image_nums_list[data_partition]
        mapping_dict = self.mapping_dicts[data_partition]
        base_dir = self.base_dirs[data_partition]
        
        # Get target record and frame
        record_num = record_nums[i]
        target_frame = image_nums[i]
        
        # Generate sequence of frame numbers
        frame_numbers = list(range(target_frame - self.context_length*2, target_frame))
        
        # Generate image paths and actions
        actions = []
        
        for frame_num in frame_numbers:
            if frame_num < 0:
                # Use padding for negative frame numbers
                x, y, left_click, right_click, key_events = 0, 0, False, False, []
                #actions.append('N N N N N N : N N N N N')
            else:
                # Use actual image and look up action
                assert (record_num, frame_num) in mapping_dict, f"No action found for record {record_num} and frame {frame_num}"
                x, y, left_click, right_click, key_events = mapping_dict.get((record_num, frame_num))
                #actions.append(self.mapping_dict.get((record_num, frame_num), 'N N N N N N : N N N N N'))
            actions.append((x, y, left_click, right_click, key_events))
        
        # Add target frame action
        assert (record_num, target_frame) in mapping_dict, f"No action found for record {record_num} and frame {target_frame}"
        actions.append(mapping_dict.get((record_num, target_frame)))
        #actions.append(self.mapping_dict.get((record_num, target_frame), 'N N N N N N : N N N N N'))
        assert len(actions) == self.context_length*2+1, f"Action sequence must be {self.context_length*2+1} actions long"
        
        # Helper function to load image using the local cache - directly takes record_num and frame_num
        def load_image_with_cache(record_num, frame_num, is_padding=False):
            # Handle padding
            if is_padding:
                padding_path = os.path.join(base_dir, 'padding.npy')
                return torch.from_numpy(np.load(padding_path))
            
            # Get tar path
            tar_path = os.path.join(base_dir, f'record_{record_num}.tar')
            key = str(frame_num)
            
            try:
                # If we don't have this tar file cached yet, load and index all its contents
                if tar_path not in local_tar_cache:
                    # Create a dictionary to store frames by key for fast lookup
                    samples_dict = {}
                    
                    # Open the tar file and read all samples
                    dataset = wds.WebDataset(tar_path).decode()
                    for sample in dataset:
                        # Store each sample in our dictionary, keyed by __key__
                        samples_dict[sample["__key__"]] = sample["npy"]
                    
                    # Cache the dictionary instead of the WebDataset object
                    local_tar_cache[tar_path] = samples_dict
                
                # Now do a direct dictionary lookup - O(1) operation
                if key in local_tar_cache[tar_path]:
                    return torch.from_numpy(local_tar_cache[tar_path][key])
                
                # Frame not found, use padding
                print(f"Warning: Frame {frame_num} not found in record {record_num}")
                padding_path = os.path.join(base_dir, 'padding.npy')
                return torch.from_numpy(np.load(padding_path))
            
            except Exception as e:
                print(f"Error loading from {tar_path}: {e}")
                padding_path = os.path.join(base_dir, 'padding.npy')
                return torch.from_numpy(np.load(padding_path))
        
        if self.use_original_image:
            assert False
            example['image'] = normalize_image(f'../data/data_processing/train_dataset/record_{record_num}/image_{target_frame}.png')
        else:
            # Load target image
            example["image_processed"] = self.normalize_features(
                load_image_with_cache(record_num, target_frame)
            )
            
            # Load context images
            context_frames = []
            for idx, frame_num in enumerate(frame_numbers):
                is_padding = frame_num < 0
                frame = load_image_with_cache(record_num, frame_num, is_padding=is_padding)
                context_frames.append(frame)
            
            example["c_concat_processed"] = self.normalize_features(torch.stack(context_frames))
            
            if self.debug_mode:
                assert False
                #print ('gere', example["image_processed"].shape)
                # draw cursor on a blank image
                white_image = np.ones((48*8, 64*8, 3)) * 255
                x, y, left_click, right_click, key_events = self.mapping_dict.get((record_num, target_frame))
                example["image"] = draw_cursor(white_image, x, y, left_click, right_click)
                # save the image
                #Image.fromarray(example["image"].astype(np.uint8)).save(f'gere_debug_image_{i}.png')
                #sys.exit(1)
        example['is_padding'] = torch.BoolTensor([frame_num < 0 for frame_num in frame_numbers])
        
        # Rest of action processing remains the same
        for j in range(self.context_length + 1):
            #example[f"action_{j}"] = actions[j:j+self.context_length+1]
            #assert len(example[f"action_{j}"]) == self.context_length+1, f"Action sequence {j} must be 8 actions long"
            #example[f"action_{j}"] = ' '.join(example[f"action_{j}"])
            #x, y, action_type = parse_action_string(actions[j+self.context_length])
            x, y, left_click, right_click, key_events = actions[j+self.context_length]
            #position_map, leftclick_map = create_position_and_click_map((x,y), action_type)
            #example[f"position_map_{j}"] = position_map
            #example[f"leftclick_map_{j}"] = leftclick_map
            example[f"x_{j}"] = torch.LongTensor([x if x is not None else 0])
            example[f"y_{j}"] = torch.LongTensor([y if y is not None else 0])
            example[f"is_leftclick_{j}"] = torch.BoolTensor([left_click])
            example[f"is_rightclick_{j}"] = torch.BoolTensor([right_click])
            example[f"key_events_{j}"] = torch.LongTensor([0 for _ in self.itos])
            for key in key_events:
                example[f"key_events_{j}"][self.stoi[key]] = 1
            

        for j in range(-1, -(self.context_length + 1), -1):
            x, y, left_click, right_click, key_events = actions[j+self.context_length]
            #position_map, leftclick_map = create_position_and_click_map((x,y), action_type)
            #example[f"position_map_{j}"] = position_map
            #example[f"leftclick_map_{j}"] = leftclick_map
            example[f"x_{j}"] = torch.LongTensor([x if x is not None else 0])
            example[f"y_{j}"] = torch.LongTensor([y if y is not None else 0])
            example[f"is_leftclick_{j}"] = torch.BoolTensor([left_click])
            example[f"is_rightclick_{j}"] = torch.BoolTensor([right_click])
            example[f"key_events_{j}"] = torch.LongTensor([0 for _ in self.itos])
            for key in key_events:
                example[f"key_events_{j}"][self.stoi[key]] = 1

        if self.normalization == 'standard_maskprev0':
            assert False
            example['c_concat_processed'] = example['c_concat_processed'] * 0
            assert False

        # Cleanup: close all tar files
        for tar in local_tar_cache.values():
            try:
                tar.close()
            except:
                pass
            
        return example 

def normalize_image(image_path): 

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
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = {}
        if train:
            self.dataset_configs["train"] = train
        if validation:
            self.dataset_configs["validation"] = validation
        if test:
            self.dataset_configs["test"] = test
        # Filter out Lightning-specific kwargs
        self.dataloader_kwargs = {k: v for k, v in kwargs.items() 
                                if k not in ['wrap']}

    def setup(self, stage=None):
        """Called by Lightning before train/val/test."""
        if not hasattr(self, 'datasets'):
            self.datasets = {}
            for k, config in self.dataset_configs.items():
                dataset = instantiate_from_config(config)
                dataset.setup()  # Call setup on the dataset
                self.datasets[k] = dataset

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], 
                         batch_size=self.batch_size,
                         **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                         batch_size=self.batch_size,
                         **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], 
                         batch_size=self.batch_size,
                         **self.dataloader_kwargs)
        
