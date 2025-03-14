import re
import matplotlib.pyplot as plt
import numpy as np

def extract_losses_from_log(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # Extract losses using regex (e.g., loss=1.88e+03)
    loss_pattern = re.compile(r'loss=([\d\.eE+-]+)')
    losses = [float(loss) for loss in loss_pattern.findall(log_content)]
    
    return losses

def smooth_losses(losses, window_size=100):
    if len(losses) < window_size:
        window_size = max(1, len(losses) // 10)
    return np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

def plot_losses(losses, smoothed_losses, output_png='loss_curve_ae_mar10_keyboard_cont.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(losses)), losses, alpha=0.3, label='Raw Loss')
    plt.plot(range(len(smoothed_losses)), smoothed_losses, color='red', linewidth=2, label='Smoothed Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve with Smoothing')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    print(f'Smoothed loss curve saved as {output_png}')

if __name__ == "__main__":
    log_file_path = 'log.ae.kl.bsz8_acc3_lr4.5e6_load_acc1.save.512_384.mar10.keyboard.cont'  # Replace with your log file path
    losses = extract_losses_from_log(log_file_path)
    smoothed_losses = smooth_losses(losses, window_size=100)
    plot_losses(losses, smoothed_losses)

