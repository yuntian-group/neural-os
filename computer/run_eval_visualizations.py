import os
import re
import fileinput
import subprocess
from pathlib import Path

# Directory containing checkpoints
ckpt_dir = 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered'

# Get all checkpoint files and sort them
ckpts = []
for f in os.listdir(ckpt_dir):
    if f.endswith('.ckpt'):
        step = int(re.search(r'step=(\d+)', f).group(1))
        ckpts.append((step, f))
ckpts.sort()  # Sort by step number

# Config file to run
config_file = 'configs/pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_difficult_only_withlstmencoder_without_norm_standard_filtered_eval.yaml'

for step, ckpt in ckpts:
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
    
    # Replace lines in ddpm.py
    ddpm_replacement = f'        exp_name = \'without_comp_norm_standard_ckpt{step}\'\n        DEBUG = True'
    
    with fileinput.FileInput('../latent_diffusion/ldm/models/diffusion/ddpm.py', inplace=True, backup='.bak') as file:
        for line in file:
            if '#### REPLACEMENT_LINE' in line:
                print(ddpm_replacement)
            else:
                print(line, end='')
    
    # Run the config twice
    for _ in range(2):
        subprocess.run(['python', 'main.py', '--config', config_file])
    
    print(f"Completed checkpoint: {ckpt}\n")

# Restore original files from backups
if os.path.exists('computer/main.py.bak'):
    os.replace('computer/main.py.bak', 'computer/main.py')
if os.path.exists('latent_diffusion/ldm/models/diffusion/ddpm.py.bak'):
    os.replace('latent_diffusion/ldm/models/diffusion/ddpm.py.bak', 'latent_diffusion/ldm/models/diffusion/ddpm.py')
