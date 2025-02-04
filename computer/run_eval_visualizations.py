import os
import re
import fileinput
import subprocess
from pathlib import Path
from tqdm import tqdm
import signal
import sys

def restore_files():
    """Restore all original files from backups"""
    if os.path.exists('./main.py.bak'):
        os.replace('./main.py.bak', './main.py')
    if os.path.exists('../latent_diffusion/ldm/models/diffusion/ddpm.py.bak'):
        os.replace('../latent_diffusion/ldm/models/diffusion/ddpm.py.bak', '../latent_diffusion/ldm/models/diffusion/ddpm.py')
    if os.path.exists(f'{config_file}.bak'):
        os.replace(f'{config_file}.bak', config_file)

# Handle Ctrl+C
def signal_handler(sig, frame):
    print('\nCtrl+C detected. Restoring files before exit...')
    restore_files()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Directory containing checkpoints
#ckpt_dir = 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered'
ckpt_dir = 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered_with_desktop_1.5k/'

# Get all checkpoint files and sort them
ckpts = []
for f in os.listdir(ckpt_dir):
    if f.endswith('.ckpt'):
        step = int(re.search(r'step=(\d+)', f).group(1))
        if step != 530000:
            continue
        ckpts.append((step, f))
ckpts.sort()  # Sort by step number

# Config file to run
config_file = 'configs/pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_difficult_only_withlstmencoder_without_norm_standard_filtered_eval.yaml'

for step, ckpt in tqdm(ckpts, desc="Processing checkpoints"):
    print(f"Processing checkpoint: {ckpt}")
    
    # Replace line in main.py
    ckpt_path = os.path.join(ckpt_dir, ckpt)
    main_replacement = f'        model = load_model_from_config(config, \'{ckpt_path}\')'
    
    with fileinput.FileInput('./main.py', inplace=True, backup='.bak') as file:
        for line in file:
            if '#### REPLACEMENT_LINE' in line:
                print(main_replacement)
            else:
                print(line, end='')
    
    # Run for training set
    # Replace lines in ddpm.py for training set
    ddpm_replacement = f'        exp_name = \'without_comp_norm_standard_ckpt{step}/train\'\n        DEBUG = True'
    
    with fileinput.FileInput('../latent_diffusion/ldm/models/diffusion/ddpm.py', inplace=True, backup='.bak') as file:
        for line in file:
            if '#### REPLACEMENT_LINE' in line:
                print(ddpm_replacement)
            else:
                print(line, end='')
    
    # Run with original config (training set)
    try:
        subprocess.run(f'python main.py --config {config_file}', shell=True)
    except Exception as e:
        print(f"Error in training run: {e}")
        pass
    
    if os.path.exists('../latent_diffusion/ldm/models/diffusion/ddpm.py.bak'):
        os.replace('../latent_diffusion/ldm/models/diffusion/ddpm.py.bak', '../latent_diffusion/ldm/models/diffusion/ddpm.py')
    
    # Now modify config for test set
    with fileinput.FileInput(config_file, inplace=True, backup='.bak') as file:
        for line in file:
            if 'data_csv_path: desktop_sequences_filtered_with_desktop_1.5k_removelast100.csv' in line:
                print('        data_csv_path: desktop_sequences_filtered_with_desktop_1.5k_last100.csv')
            else:
                print(line, end='')
    
    # Replace lines in ddpm.py for test set
    ddpm_replacement = f'        exp_name = \'without_comp_norm_standard_ckpt{step}/test\'\n        DEBUG = True'
    
    with fileinput.FileInput('../latent_diffusion/ldm/models/diffusion/ddpm.py', inplace=True, backup='.bak') as file:
        for line in file:
            if '#### REPLACEMENT_LINE' in line:
                print(ddpm_replacement)
            else:
                print(line, end='')
    
    # Run with modified config (test set)
    try:
        subprocess.run(f'python main.py --config {config_file}', shell=True)
    except Exception as e:
        print(f"Error in test run: {e}")
        pass
    
    # Restore original files
    restore_files()
    
    print(f"Completed checkpoint: {ckpt}\n")

# Final cleanup
restore_files()
