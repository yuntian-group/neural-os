from typing import Dict, List, Tuple, Union, Any
from PIL import Image
import random
import torch
import numpy as np
import argparse
from omegaconf import OmegaConf
import time
import os
from pathlib import Path
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion, DDIMSampler
from computer.util import load_cond_from_config, load_first_stage_from_config, load_model, load_model_from_config, get_ground_truths, init_model, load_autoencoder_from_ckpt, load_cond_from_ckpt

# Configuration constants
SCREEN_WIDTH = 512
SCREEN_HEIGHT = 384
LATENT_DIMS = [4, 48, 64]
NUM_SAMPLING_STEPS = 8
DATA_NORMALIZATION = {
    'mean': -0.54,
    'std': 6.78,
    'min': -27.681446075439453,
    'max': 30.854148864746094
}

# Valid keyboard inputs
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
INVALID_KEYS = ['f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20',
                'f21', 'f22', 'f23', 'f24', 'select', 'separator', 'execute']
VALID_KEYS = [key for key in KEYS if key not in INVALID_KEYS]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

##Parse args here

def generate_input_sample() -> Tuple[int, int, bool, bool, str, Dict[str, int], List[str]]:
    """Generate a random input sample for the model."""
    x = random.randint(0, SCREEN_WIDTH - 1)
    y = random.randint(0, SCREEN_HEIGHT - 1)
    right_click = bool(random.randint(0, 1))
    left_click = bool(random.randint(0, 1))
    
    key = random.choice(VALID_KEYS)
    stoi = {k: i for i, k in enumerate(VALID_KEYS)}
    
    return x, y, right_click, left_click, key, stoi, VALID_KEYS

def prepare_model_inputs(
    previous_frame: torch.Tensor,
    hidden_states: Any,
    x: int,
    y: int,
    right_click: bool,
    left_click: bool,
    key: str,
    stoi: Dict[str, int],
    itos: List[str],
    time_step: int
) -> Dict[str, torch.Tensor]:
    """Prepare inputs for the model."""
    inputs = {
        'image_features': previous_frame.to(device),
        'is_padding': torch.BoolTensor([time_step == 0]).to(device),
        'x': torch.LongTensor([x if x is not None else 0]).unsqueeze(0).to(device),
        'y': torch.LongTensor([y if y is not None else 0]).unsqueeze(0).to(device),
        'is_leftclick': torch.BoolTensor([left_click]).unsqueeze(0).to(device),
        'is_rightclick': torch.BoolTensor([right_click]).unsqueeze(0).to(device),
        'key_events': torch.zeros(len(itos), dtype=torch.long).to(device)
    }
    
    inputs['key_events'][stoi[key]] = 1
    
    if hidden_states is not None:
        inputs['hidden_states'] = hidden_states
    
    return inputs

@torch.no_grad()
def process_frame(
    model: LatentDiffusion,
    inputs: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, np.ndarray, Any, Dict[str, float]]:
    """Process a single frame through the model."""
    timing = {}
    
    # Temporal encoding
    start = time.perf_counter()
    output_from_rnn, hidden_states = model.temporal_encoder.forward_step(inputs)
    timing['temporal_encoder'] = time.perf_counter() - start
    
    # UNet sampling
    start = time.perf_counter()
    sampler = DDIMSampler(model)
    sample_latent, _ = sampler.sample(
        S=NUM_SAMPLING_STEPS,
        conditioning={'c_concat': output_from_rnn},
        batch_size=1,
        shape=LATENT_DIMS,
        verbose=False
    )
    timing['unet'] = time.perf_counter() - start
    
    # Decoding
    start = time.perf_counter()
    sample = sample_latent * DATA_NORMALIZATION['std'] + DATA_NORMALIZATION['mean']
    sample = model.decode_first_stage(sample)
    sample = sample.squeeze(0).clamp(-1, 1)
    timing['decode'] = time.perf_counter() - start
    
    # Convert to image
    sample_img = ((sample[:3].transpose(0,1).transpose(1,2).cpu().float().numpy() + 1) * 127.5).astype(np.uint8)
    
    timing['total'] = sum(timing.values())
    
    return sample_latent, sample_img, hidden_states, timing

def save_frame(img: np.ndarray, frame_num: int, output_dir: str = 'config_exp'):
    """Save a frame to disk."""
    Path(output_dir).mkdir(exist_ok=True)
    Image.fromarray(img).save(f'{output_dir}/sample_img_{frame_num}.png')

def print_timing_stats(timing_info: Dict[str, float], frame_num: int):
    """Print timing statistics for a frame."""
    print(f"\nFrame {frame_num} timing (seconds):")
    for key, value in timing_info.items():
        print(f"  {key.title()}: {value:.4f}")
    print(f"  FPS: {1.0/timing_info['total']:.2f}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train and sample a model using a config file')
    parser.add_argument('--config', 
                       default="configs/standard_challenging_context32_nocond_all_rnn.eval.yaml",
                       help='Path to the configuration file')
    args = parser.parse_args()

    # Load configuration and model
    config = OmegaConf.load(args.config)
    save_path = config.save_path
    print ('='*10)
    print (save_path)
    print (args.config)
    #print (config.model.scheduler_sampling_rate)
    print ('='*10)
    #import pdb; pdb.set_trace()
    #from_autoencoder = True
    from_scratch = False
    #from_autoencoder = False # TODO: fix
    if from_scratch:
        model = init_model(config)
        #model = load_first_stage_from_config(model, './autoencoder_saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_model-603000.ckpt')
        model = load_first_stage_from_config(model, '../autoencoder/saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384/model-354000.ckpt')
    else:
        #model = load_model_from_config(config, './saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered_with_desktop_1.5k_eval/model_saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered_with_desktop_1.5k_eval.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond/model-step=720000.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond_fixnorm_all/model-step=308000.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2/model-step=024000.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn_fixrnn/model-step=004000.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn_fixrnn_enablegrad_all_keyevent_cont/model-step=006000.ckpt')
        model = load_model_from_config(config, 'saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn_fixrnn_enablegrad_all_keyevent_cont_clusters/model-step=030000.ckpt')

    # Move model to GPU
    model = model.to(device)
    
    num_frames = 2000
    previous_frame = torch.zeros(1, 48, 64, 4).to(device)
    hidden_states = None
    all_timings = []

    # Process frames
    for frame_num in range(num_frames):
        print(f"\nProcessing frame {frame_num}")
        
        # Generate input
        x, y, right_click, left_click, key, stoi, itos = generate_input_sample()
        
        # Process frame
        start_frame = time.perf_counter()
        inputs = prepare_model_inputs(
            previous_frame, hidden_states, x, y, right_click, left_click, 
            key, stoi, itos, frame_num
        )
        previous_frame, sample_img, hidden_states, timing_info = process_frame(model, inputs)
        timing_info['full_frame'] = time.perf_counter() - start_frame
        
        # Save results
        save_frame(sample_img, frame_num)
        print_timing_stats(timing_info, frame_num)
        all_timings.append(timing_info)

    # Print summary statistics
    print("\nSummary Statistics:")
    timing_keys = ['temporal_encoder', 'unet', 'decode', 'total', 'full_frame']
    timings_array = np.array([[t[k] for k in timing_keys] for t in all_timings])
    means = np.mean(timings_array, axis=0)
    stds = np.std(timings_array, axis=0)
    
    for key, mean, std in zip(timing_keys, means, stds):
        print(f"  {key.title()}: {mean:.4f} ± {std:.4f} seconds")
    print(f"Average FPS: {1.0/means[-1]:.2f}")

if __name__ == "__main__":
    main()


