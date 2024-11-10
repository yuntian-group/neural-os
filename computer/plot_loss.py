import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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

# Plot each log file
for log_file in glob.glob("log.pssearch_*"):
    if '4e4' in log_file:
        continue
    if '8e4' in log_file:
        continue
    if '2e4' in log_file:
        continue
    if '1e4' in log_file:
        continue
    # Extract learning rate and accumulation from filename
    params = log_file.split('_')
    lr = params[-1]
    acc = params[-2].replace('acc', '')
    label = f'lr={lr}, acc={acc}'
    
    steps, losses = extract_losses(log_file)
    plt.plot(steps, losses, label=label, alpha=0.8)
    plt.ylim(0, 0.05)

plt.xlabel('Global Step')
plt.ylabel('Average Loss')
plt.title('Training Loss Curves')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  # Adjust layout to prevent legend cutoff

# Save plot
plt.savefig('loss_curves_comparison.png', bbox_inches='tight')
print(f"Plot saved to loss_curves_comparison.png")
