import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Dictionary to store losses for each step
step_losses = defaultdict(list)

# Read and parse log file
with open(sys.argv[1], 'r') as f:
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

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(steps, avg_losses)
plt.xlabel('Global Step')
plt.ylabel('Average Loss')
plt.title('Training Loss Curve')
plt.grid(True)

# Save plot
plt.savefig(sys.argv[1] + '.png')
print(f"Plot saved to {sys.argv[1]}.png")
