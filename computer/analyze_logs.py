import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_log_file(file_path):
    """Parse log file and extract loss values"""
    losses = []
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Split on Epoch markers to handle the continuous tqdm output
    epochs = content.split('Epoch ')
    
    for epoch_data in epochs[1:]:  # Skip first split as it's empty
        # Find all loss values in this epoch
        matches = re.finditer(r'loss=(\d+\.\d+)', epoch_data)
        for match in matches:
            losses.append(float(match.group(1)))
    
    return np.array(losses)

def smooth_curve(y, window_size=100):
    """Smooth the curve using moving average"""
    # Ensure window_size is not larger than data
    window_size = min(window_size, len(y))
    
    # Calculate moving average
    weights = np.ones(window_size) / window_size
    smoothed_y = np.convolve(y, weights, mode='valid')
    
    return smoothed_y

def plot_training_curves(log_files, window_size=100):
    """Plot smoothed training curves for multiple log files"""
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Color palette for different curves
    colors = sns.color_palette("husl", len(log_files))
    
    for file_path, color in zip(log_files, colors):
        # Extract experiment name from file path
        exp_name = os.path.basename(file_path).replace('log.pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_difficult_only_withlstmencoder_', '')
        
        # Parse log file
        losses = parse_log_file(file_path)
        
        # Smooth the losses
        smoothed_losses = smooth_curve(losses, window_size)
        
        # Create steps array
        steps = np.arange(len(smoothed_losses))
        
        # Plot
        plt.plot(steps, smoothed_losses, label=exp_name, color=color, alpha=0.8)
    
    plt.xlabel('Steps (hundreds)')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves (Smoothed)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('training_curves_comparison.png', dpi=300, bbox_inches='tight')
    
    # Print some statistics
    print("\nTraining Statistics:")
    print("-" * 50)
    for file_path in log_files:
        exp_name = os.path.basename(file_path).replace('log.pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_difficult_only_withlstmencoder_', '')
        losses = parse_log_file(file_path)
        print(f"\n{exp_name}:{file_path}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Min loss: {np.min(losses):.4f}")
        print(f"Mean loss: {np.mean(losses):.4f}")
        print(f"Total iterations: {len(losses)}")

if __name__ == "__main__":
    log_files = [
        "log.pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_difficult_only_withlstmencoder_2048",
        "log.pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_difficult_only_withlstmencoder_4096",
        "log.pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_difficult_only_withlstmencoder_8192",
        "log.pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_difficult_only_withlstmencoder_8192_1layer",
        "log.pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_difficult_only_withlstmencoder_8192_1layer_trim",
        "log.pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_difficult_only_withlstmencoder_without"
    ]
    
    plot_training_curves(log_files, window_size=100)
