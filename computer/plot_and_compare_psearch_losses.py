import re
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import datetime

def elapsed_time_since_modified(file_path):
    """Returns the elapsed time since the file was last modified."""
    try:
        modification_time = os.path.getctime(file_path)
        current_time = datetime.datetime.now().timestamp()
        #current_time = os.path.getmtime(file_path)
        elapsed_seconds = current_time - modification_time
        return elapsed_seconds #datetime.timedelta(seconds=elapsed_seconds)
    except FileNotFoundError:
        return None



def extract_losses_from_log(log_file_path):
    log_content = ''
    if '.cont2' in log_file_path:
        with open(log_file_path.replace('.cont2', ''), 'r') as file:
            log_content += file.read()
        with open(log_file_path.replace('.cont2', '.cont'), 'r') as file:
            log_content += file.read()
    elif '.cont' in log_file_path:
        with open(log_file_path.replace('.cont', ''), 'r') as file:
            log_content += file.read()
    with open(log_file_path, 'r') as file:
        log_content += file.read()

    # Extract losses using regex (e.g., loss=1.88e+03)
    loss_pattern = re.compile(r'loss_step=([\d\.eE+-]+)')
    losses = [math.log(float(loss)) for loss in loss_pattern.findall(log_content)]
    print (len(losses))
    
    return losses

def smooth_losses(losses, window_size=100):
    import pandas as pd
    series = pd.Series(losses)
    return series.rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()
    #if len(losses) < window_size:
    #    window_size = max(1, len(losses) // 10)
    #return np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

def plot_and_compare_losses(losses_dict, output_png='psearch_b_loss_curve_comparison.png'):
    """
    Plot and compare multiple loss curves on a single graph.
    
    :param losses_dict: A dictionary where keys are labels and values are tuples of (losses, smoothed_losses).
    :param output_png: The filename for the output plot image.
    """
    plt.figure(figsize=(12, 6))
    
    for label, (losses, smoothed_losses) in losses_dict.items():
        #plt.plot(range(len(losses)), losses, alpha=0.3, label=f'{label} Raw Loss')
        plt.plot(range(len(smoothed_losses)), smoothed_losses, linewidth=2, label=f'{label} Smoothed Loss')
        #plt.plot(range(len(losses)), losses, linewidth=2, label=f'{label} Smoothed Loss')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve Comparison')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    print(f'Loss curve comparison saved as {output_png}')
#

if __name__ == "__main__":
    log_file_paths = [
        #('a_hs4096_oc32_nl48_ar_cm1_2_mc320', 'log.a_hs4096_oc32_nl48_ar_cm1_2_mc320'),
        ('a_hs4096_oc32_nl48_ar_cm1_2_mc384', 'log.a_hs4096_oc32_nl48_ar_cm1_2_mc384'),
        #('a_hs4096_oc32_nl48_ar2_cm1_2_mc320', 'log.a_hs4096_oc32_nl48_ar2_cm1_2_mc320'),
        #('a_hs4096_oc32_nl48_ar_cm1_2_3_mc320', 'log.a_hs4096_oc32_nl48_ar_cm1_2_3_mc320'),
        #('a_hs4096_oc32_nl48_ar2_cm1_2_3_mc320', 'log.a_hs4096_oc32_nl48_ar2_cm1_2_3_mc320'),
        #('a_hs4096_oc32_nl48_ar4_cm1_2_3_mc320', 'log.a_hs4096_oc32_nl48_ar4_cm1_2_3_mc320'),
        #('a_hs1024_oc4_nl20_ar2_4_8_cm1_2_3_5_mc192', 'log.a_hs1024_oc4_nl20_ar2_4_8_cm1_2_3_5_mc192'),
        #('a_hs4096_oc32_nl48_ar2_4_8_cm1_2_3_5_mc320', 'log.a_hs4096_oc32_nl48_ar2_4_8_cm1_2_3_5_mc320'),
        #('b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b64', 'log.b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b64'),
        #('b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr4e5_b64', 'log.b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr4e5_b64'),
        #('b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr1.6e4_b64', 'log.b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr1.6e4_b64'),
        #('b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b128', 'log.b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b128'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr4e5_b64', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr4e5_b64'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr1.6e4_b64', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr1.6e4_b64'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b100', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b100'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu1', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu1'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu2', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu2'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu4', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu4'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8'),
        ('orig', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.cont'),
        #('final', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.cont'),
        #('final1', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg'),
        #('final2', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.bsz50'),
        #('final20', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5'),
        #('final3', 'log.final_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b64_gpu8_filtered.largeimg.lr4e5'),
        ('largeimg', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.cont2'),
        #('1gpu', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.gpu1'),
        #('2gpu', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.gpu2'),
        #('4gpu', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.gpu4'),
    ]
    
    losses_dict = {}
    for label, log_file_path in log_file_paths:
        losses = extract_losses_from_log(log_file_path)
        smoothed_losses = smooth_losses(losses, window_size=500)
        time_elapsed = elapsed_time_since_modified(log_file_path)
        if 'b128' in label:
            time_elapsed = 0.5
        elif 'b100' in label:
            time_elapsed = 64/100
        else:
            time_elapsed = 1
        print (label, len(losses), len(losses) / time_elapsed, smoothed_losses[-1])
        #losses_dict[label] = (losses[:24000], smoothed_losses[:24000])
        losses_dict[label] = (losses[:256000], smoothed_losses[:256000])
        #losses_dict[label] = (losses[:6000], smoothed_losses[:6000])
    
    plot_and_compare_losses(losses_dict)

