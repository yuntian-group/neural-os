import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

def smooth_curve(y, sigma=2):
    """Apply Gaussian smoothing to the curve"""
    return gaussian_filter1d(y, sigma=sigma)

def moving_average(y, window=100):
    """Calculate moving average with given window size"""
    return np.convolve(y, np.ones(window), 'valid') / window

def extract_losses(filename):
    step_losses = defaultdict(list)
    
    with open(filename, 'r') as f:
        for line in f:
            # Find all occurrences of global_step and loss in the line
            step_matches = re.finditer(r'global_step=(\d+)', line)
            loss_matches = re.finditer(r'train/loss_step=(\d+\.\d+)', line)
            
            # Convert iterators to lists
            steps = [int(m.group(1)) for m in step_matches]
            losses = [float(m.group(1)) for m in loss_matches]
            
            # Add all losses to their corresponding steps
            for step, loss in zip(steps, losses):
                step_losses[step].append(loss)
    
    # Calculate average loss for each step
    steps = sorted(step_losses.keys())
    avg_losses = [np.mean(step_losses[step]) for step in steps]
    
    return steps, avg_losses

# Create the plot
plt.figure(figsize=(12, 8))

# Store final statistics
final_stats = []

# Plot each log file
for log_file in glob.glob("log.pssearch_*"):
    # Extract learning rate and accumulation from filename
    params = log_file.split('_')
    lr = params[-1]
    acc = params[-2].replace('acc', '')
    
    steps, losses = extract_losses(log_file)
    
    # Apply smoothing
    smooth_losses = smooth_curve(losses)
    
    # Calculate moving average for final 1000 steps
    final_ma = moving_average(losses[-1000:] if len(losses) > 1000 else losses)
    final_stats.append({
        'lr': lr,
        'acc': acc,
        'final_loss_ma': final_ma.mean(),
        'min_loss': min(smooth_losses)
    })
    
    label = f'lr={lr}, acc={acc}'
    plt.plot(steps, smooth_losses, label=label, alpha=0.8)

plt.xlabel('Global Step')
plt.ylabel('Average Loss (Smoothed)')
plt.title('Training Loss Curves (Gaussian Smoothed)')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save plot
plt.savefig('loss_curves_comparison.png', bbox_inches='tight')
print(f"Plot saved to loss_curves_comparison.png")

# Print statistics sorted by final moving average
print("\nFinal Statistics (sorted by final moving average):")
print("=" * 60)
print(f"{'LR':<10} {'Acc':<6} {'Final MA':<12} {'Min Loss':<12}")
print("-" * 60)
for stat in sorted(final_stats, key=lambda x: x['final_loss_ma']):
    print(f"{stat['lr']:<10} {stat['acc']:<6} {stat['final_loss_ma']:.6f} {stat['min_loss']:.6f}") 