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

def plot_and_compare_losses(losses_dict, output_png='loss_curve_comparison.cont.png'):
    """
    Plot and compare multiple loss curves on a single graph.
    
    :param losses_dict: A dictionary where keys are labels and values are tuples of (losses, smoothed_losses).
    :param output_png: The filename for the output plot image.
    """
    plt.figure(figsize=(12, 6))
    
    for label, (losses, smoothed_losses) in losses_dict.items():
        #plt.plot(range(len(losses)), losses, alpha=0.3, label=f'{label} Raw Loss')
        plt.plot(range(len(smoothed_losses)), smoothed_losses, linewidth=2, label=f'{label} Smoothed Loss')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve Comparison')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    print(f'Loss curve comparison saved as {output_png}')

if __name__ == "__main__":
    log_file_paths = [
        ('init_4', 'log.init.ae.kl.bsz8_acc3_lr4.5e6_load_acc1.save.512_384.mar10.keyboard.4.cont'),
        ('init_8', 'log.init.ae.kl.bsz8_acc3_lr4.5e6_load_acc1.save.512_384.mar10.keyboard.8.cont'),
        ('init_16', 'log.init.ae.kl.bsz8_acc3_lr4.5e6_load_acc1.save.512_384.mar10.keyboard.16.cont'),
        ('cont', 'log.ae.kl.bsz8_acc3_lr4.5e6_load_acc1.save.512_384.mar10.keyboard.cont2.cont')
    ]
    
    losses_dict = {}
    for label, log_file_path in log_file_paths:
        losses = extract_losses_from_log(log_file_path)
        smoothed_losses = smooth_losses(losses, window_size=100)
        losses_dict[label] = (losses, smoothed_losses)
    
    plot_and_compare_losses(losses_dict)

