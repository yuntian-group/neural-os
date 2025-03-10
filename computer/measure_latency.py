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
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
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

def create_dummy_model(config):
    """Create a dummy model with randomly initialized weights."""
    model = init_model(config)
    
    # # Simple random initialization for all parameters
    # def init_random_weights(m):
    #     for param in m.parameters():
    #         if param.dim() > 1:
    #             # For weight matrices (2D+)
    #             torch.nn.init.xavier_normal_(param.data)
    #         else:
    #             # For bias vectors and 1D parameters
    #             torch.nn.init.normal_(param.data, 0.0, 0.02)
    
    # # Apply the random initialization to all modules
    # model.apply(init_random_weights)
    
    model.eval()  # Set to evaluation mode
    return model

def save_results_to_file(results, output_dir='.'):
    """Save detailed results to a text file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename based on config name
    safe_name = results['config_name'].replace('/', '_').replace('\\', '_')
    file_path = os.path.join(output_dir, f"results_{safe_name}.txt")
    
    with open(file_path, 'w') as f:
        # Write header with config name and top-level metrics
        f.write(f"Results for: {results['config_name']}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"FPS: {results['fps']:.2f}\n")
        f.write(f"Total frames processed: {results['total_frames']}\n")
        f.write(f"Warmup frames skipped from statistics: {results['warmup_frames']}\n\n")
        
        # Write component-wise timing summary
        f.write("Component Timing Summary (in milliseconds):\n")
        f.write(f"{'-' * 40}\n")
        f.write(f"{'Component':<20} {'Mean':>10} {'Std Dev':>10}\n")
        
        for component in ['temporal_encoder', 'unet', 'decode', 'full_frame']:
            mean_ms = results['means'][component] * 1000
            std_ms = results['stds'][component] * 1000
            f.write(f"{component:<20} {mean_ms:>10.2f} {std_ms:>10.2f}\n")
        
        # Write frame-by-frame timings
        f.write("\nFrame-by-Frame Timings (in milliseconds, after warmup):\n")
        f.write(f"{'-' * 70}\n")
        f.write(f"{'Frame':>6} {'Temporal':>12} {'UNet':>12} {'Decode':>12} {'Total':>12}\n")
        
        for i, timing in enumerate(results['all_timings']):
            actual_frame = i + results['warmup_frames']  # Adjust frame number to account for warmup
            f.write(f"{actual_frame:>6} {timing['temporal_encoder']*1000:>12.2f} {timing['unet']*1000:>12.2f} "
                    f"{timing['decode']*1000:>12.2f} {timing['full_frame']*1000:>12.2f}\n")
    
    print(f"Detailed results saved to: {file_path}")
    return file_path

def process_config(config_path, num_frames=10):
    """Process a single configuration file."""
    print(f"\n{'='*50}")
    print(f"Processing config: {config_path}")
    print(f"{'='*50}")
    
    # Load configuration
    config = OmegaConf.load(config_path)
    config_name = Path(config_path).stem
    
    # Create dummy model
    model = create_dummy_model(config)
    model = model.to(device)
    
    previous_frame = torch.zeros(1, 48, 64, 4).to(device)
    hidden_states = None
    all_timings = []
    
    # Define number of warmup frames to skip in statistics
    warmup_frames = 5
    print(f"Processing {num_frames} frames (first {warmup_frames} frames will be excluded from statistics)")

    # Process frames
    for frame_num in range(num_frames):
        if frame_num % 20 == 0:
            print(f"Processing frame {frame_num}/{num_frames}")
        
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
        
        # Only add timing info after warmup period
        if frame_num >= warmup_frames:
            all_timings.append(timing_info)

    # Calculate metrics (now only on frames after warmup)
    means = {k: np.mean([t[k] for t in all_timings]) for k in all_timings[0].keys()}
    stds = {k: np.std([t[k] for t in all_timings]) for k in all_timings[0].keys()}
    
    # Calculate FPS based on full frame time
    fps = 1.0 / means['full_frame']
    
    # Create results dictionary
    results = {
        'config_name': Path(config_path).stem,
        'fps': fps,
        'means': means,
        'stds': stds,
        'all_timings': all_timings,
        'warmup_frames': warmup_frames,
        'total_frames': num_frames
    }
    
    # Print summary
    print(f"\nResults for {results['config_name']}:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Component timing (mean ± std) after {warmup_frames} warmup frames:")
    for component in ['temporal_encoder', 'unet', 'decode']:
        print(f"    {component}: {means[component]*1000:.2f} ± {stds[component]*1000:.2f} ms")
    print(f"  Total: {means['full_frame']*1000:.2f} ± {stds['full_frame']*1000:.2f} ms")
    
    # Save detailed results to file
    save_results_to_file(results, output_dir="latency_results")
    
    return results

def plot_comparison(results):
    """Create comparison plots across configurations."""
    
    # 1. Bar chart of average latency components (sort by total latency)
    # Sort results by total latency (ascending - lower is better)
    sorted_results_by_latency = sorted(results, key=lambda r: r['means']['full_frame'])
    config_names_latency = [r['config_name'] for r in sorted_results_by_latency]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    components = ['temporal_encoder', 'unet', 'decode']
    x = np.arange(len(config_names_latency))
    width = 0.25
    
    for i, component in enumerate(components):
        means = [r['means'][component] * 1000 for r in sorted_results_by_latency]  # Convert to ms
        stds = [r['stds'][component] * 1000 for r in sorted_results_by_latency]    # Convert to ms
        bars = ax.bar(x + i*width - width, means, width, label=component.replace('_', ' ').title(), 
               yerr=stds, capsize=5)
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stds[j] + 1,
                    f'{means[j]:.1f}ms',
                    ha='center', va='bottom', rotation=0, fontsize=8)
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Component Latency Comparison (Sorted by Total Latency)')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names_latency, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig('component_latency_comparison.png')
    
    # 2. Total latency comparison (sorted by total latency)
    fig, ax = plt.subplots(figsize=(10, 6))
    total_means = [r['means']['full_frame'] * 1000 for r in sorted_results_by_latency]  # Convert to ms
    total_stds = [r['stds']['full_frame'] * 1000 for r in sorted_results_by_latency]    # Convert to ms
    
    bars = ax.bar(config_names_latency, total_means, yerr=total_stds, capsize=5)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + total_stds[i] + 1,
                f'{total_means[i]:.1f}ms',
                ha='center', va='bottom', rotation=0)
    
    ax.set_ylabel('Total Latency (ms)')
    ax.set_title('Total Inference Latency (Sorted from Lowest to Highest)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('total_latency_comparison.png')
    
    # 3. FPS comparison (sorted by FPS)
    # Sort results by FPS (descending - higher is better)
    sorted_results_by_fps = sorted(results, key=lambda r: r['fps'], reverse=True)
    config_names_fps = [r['config_name'] for r in sorted_results_by_fps]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fps_values = [r['fps'] for r in sorted_results_by_fps]
    
    bars = ax.bar(config_names_fps, fps_values)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{fps_values[i]:.1f}',
                ha='center', va='bottom', rotation=0)
    
    ax.set_ylabel('Frames Per Second')
    ax.set_title('Inference Speed Comparison (Sorted from Highest to Lowest FPS)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('fps_comparison.png')
    
    # 4. Time series plot of frame latencies (use original order for consistent colors)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, result in enumerate(results):
        frame_times = [t['full_frame'] * 1000 for t in result['all_timings']]  # Convert to ms
        ax.plot(range(len(frame_times)), frame_times, label=result['config_name'])
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Frame Processing Time (ms)')
    ax.set_title('Frame-by-Frame Latency')
    ax.legend()
    plt.tight_layout()
    plt.savefig('frame_latency_timeseries.png')
    
    print("\nPlots saved to current directory.")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Measure latency across multiple model configurations')
    parser.add_argument('--configs', nargs='+', 
                       default=["computer/configs/standard_challenging_context32_nocond_all_rnn.eval.yaml"],
                       help='List of configuration files to test')
    parser.add_argument('--frames', type=int, default=200,
                       help='Number of frames to process per config')
    parser.add_argument('--output-dir', default='latency_results',
                       help='Directory to save results')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    summary_paths = []
    
    # Process each config
    for config_path in tqdm(args.configs, desc="Processing configs"):
        results = process_config(config_path, num_frames=args.frames)
        all_results.append(results)
        # Save individual results to file
        summary_path = save_results_to_file(results, output_dir=args.output_dir)
        summary_paths.append(summary_path)
    
    # Save summary comparison data
    summary_data = {
        'configs': [r['config_name'] for r in all_results],
        'fps': [r['fps'] for r in all_results],
        'total_latency_ms': [r['means']['full_frame'] * 1000 for r in all_results],
    }
    
    # Write overall summary file
    summary_file = os.path.join(args.output_dir, "summary_comparison.txt")
    with open(summary_file, 'w') as f:
        f.write("Overall Comparison Summary\n")
        f.write("=========================\n\n")
        f.write("Configurations tested:\n")
        for i, config in enumerate(summary_data['configs']):
            f.write(f"{i+1}. {config}: {summary_data['fps'][i]:.2f} FPS, {summary_data['total_latency_ms'][i]:.2f} ms latency\n")
        
        # Add sorted results by FPS
        sorted_indices = sorted(range(len(summary_data['fps'])), key=lambda i: summary_data['fps'][i], reverse=True)
        f.write("\nConfigurations Ranked by FPS (highest to lowest):\n")
        for i, idx in enumerate(sorted_indices):
            config = summary_data['configs'][idx]
            f.write(f"{i+1}. {config}: {summary_data['fps'][idx]:.2f} FPS, {summary_data['total_latency_ms'][idx]:.2f} ms latency\n")
    
    print(f"\nSummary Comparison:")
    for i, config in enumerate(summary_data['configs']):
        print(f"  {config}: {summary_data['fps'][i]:.2f} FPS, {summary_data['total_latency_ms'][i]:.2f} ms latency")
    print(f"\nDetailed comparison saved to: {summary_file}")
    
    # Create comparison plots
    if len(all_results) > 1:
        plot_comparison(all_results)
        print(f"Plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()



