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
ckpt_dir = 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered_with_desktop_1.5k_maskprev0/'
ckpt_dir = 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered_with_desktop_1.5k_maskprev0_challenging/'
ckpt_dir = 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered_with_desktop_1.5k_maskprev0_challenging_standard/'

#for context_length in [2, 4, 8, 16, 32, 64]:
#for context_length in [4, 8, 16, 32]:
#for context_length in [77]:
# 8 settings:
# 1. a_hs4096_oc32_nl48_ar_cm1_2_mc320: 8.47 FPS, 118.13 ms latency
# 2. a_hs4096_oc32_nl48_ar_cm1_2_mc384: 8.18 FPS, 122.19 ms latency
# 5. a_hs4096_oc32_nl48_ar2_cm1_2_mc320: 7.22 FPS, 138.49 ms latency
# 6. a_hs4096_oc32_nl48_ar_cm1_2_3_mc320: 7.16 FPS, 139.64 ms latency
# 13. a_hs4096_oc32_nl48_ar2_cm1_2_3_mc320: 6.25 FPS, 159.98 ms latency
# 14. a_hs4096_oc32_nl48_ar4_cm1_2_3_mc320: 6.24 FPS, 160.33 ms latency
# 34. a_hs1024_oc4_nl20_ar2_4_8_cm1_2_3_5_mc192: 4.84 FPS, 206.49 ms latency
# 41. a_hs4096_oc32_nl48_ar2_4_8_cm1_2_3_5_mc320: 4.50 FPS, 222.29 ms latency
for setting in ['a_hs4096_oc32_nl48_ar_cm1_2_mc320', 'a_hs4096_oc32_nl48_ar_cm1_2_mc384', 'a_hs4096_oc32_nl48_ar2_cm1_2_mc320', 'a_hs4096_oc32_nl48_ar_cm1_2_3_mc320', 'a_hs4096_oc32_nl48_ar2_cm1_2_3_mc320', 'a_hs4096_oc32_nl48_ar4_cm1_2_3_mc320', 'a_hs1024_oc4_nl20_ar2_4_8_cm1_2_3_5_mc192', 'a_hs4096_oc32_nl48_ar2_4_8_cm1_2_3_5_mc320']:
#for setting in ['a_hs4096_oc32_nl48_ar2_cm1_2_mc320']:
#for setting in ['a_hs1024_oc4_nl20_ar2_4_8_cm1_2_3_5_mc192']:#, 'a_hs4096_oc32_nl48_ar_cm1_2_mc384', 'a_hs4096_oc32_nl48_ar2_cm1_2_mc320', 'a_hs4096_oc32_nl48_ar_cm1_2_3_mc320', 'a_hs4096_oc32_nl48_ar2_cm1_2_3_mc320', 'a_hs4096_oc32_nl48_ar4_cm1_2_3_mc320', 'a_hs1024_oc4_nl20_ar2_4_8_cm1_2_3_5_mc192', 'a_hs4096_oc32_nl48_ar2_4_8_cm1_2_3_5_mc320']:
    #ckpt_dir = f'saved_standard_challenging_context{context_length}'
    #ckpt_dir = f'saved_standard_challenging_context32_nocond_cont_cont_all_cont/'
    #ckpt_dir = f'saved_standard_challenging_context32_nocond_fixnorm_all/'
    #ckpt_dir = f'saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1/'
    #ckpt_dir = f'saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn/'
    #ckpt_dir = f'saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn_fixrnn_enablegrad_all_keyevent_cont/'
    #ckpt_dir = f'saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn_fixrnn_enablegrad_all_keyevent_cont_clusters/'
    ckpt_dir = f'saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn_fixrnn_enablegrad_all_keyevent_{setting}/'
    print ('='*10)
    print (f'processing setting {setting}')
    # Get all checkpoint files and sort them
    ckpts = []
    for f in os.listdir(ckpt_dir):
        if f.endswith('.ckpt'):
            step = int(re.search(r'step=(\d+)', f).group(1))
            #if step != 46000:
            #    continue
            ckpts.append((step, f))
    ckpts.sort()  # Sort by step number
    ckpts = [ckpts[-1]]
    
    # Config file to run
    #config_file = f'configs/standard_challenging_context{context_length}.eval.yaml'
    #config_file = f'configs/standard_challenging_context32_nocond_all.eval.yaml'
    #config_file = f'configs/standard_challenging_context32_nocond_all_rnn.eval.yaml'
    config_file = f'configs/psearch_{setting}.eval.yaml'
    #config_file = f'configs/standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn_fixrnn_enablegrad_all_keyevent_{setting}.eval.yaml'
    
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
        ddpm_replacement = f'        exp_name = \'psearch_a_vis_norm_standard_context{setting}_ckpt{step}/train\'\n        DEBUG = True'
        
        with fileinput.FileInput('../latent_diffusion/ldm/models/diffusion/ddpm.py', inplace=True, backup='.bak') as file:
            for line in file:
                if '#### REPLACEMENT_LINE' in line:
                    print(ddpm_replacement)
                else:
                    print(line, end='')
        
        # Run with original config (training set)
        # Now modify config for test set
        with fileinput.FileInput(config_file, inplace=True, backup='.bak') as file:
            for line in file:
                if 'data_csv_path' in line:
                    #print('        data_csv_path: desktop_sequences_filtered_with_desktop_1.5k_last100.csv')
                    print('        data_csv_path: ../data/data_processing/train_dataset/filtered_dataset.target_frames.clustered.train_shuffled.shuffled.csv')
                else:
                    print(line, end='')
        
        try:
            subprocess.run(f'CUDA_VISIBLE_DEVICES=6 python main.py --config {config_file}', shell=True)
        except Exception as e:
            print(f"Error in training run: {e}")
            pass
        
        if os.path.exists('../latent_diffusion/ldm/models/diffusion/ddpm.py.bak'):
            os.replace('../latent_diffusion/ldm/models/diffusion/ddpm.py.bak', '../latent_diffusion/ldm/models/diffusion/ddpm.py')
        
        # Now modify config for test set
        with fileinput.FileInput(config_file, inplace=True, backup='.bak') as file:
            for line in file:
                if 'data_csv_path' in line:
                    #print('        data_csv_path: desktop_sequences_filtered_with_desktop_1.5k_last100.csv')
                    print('        data_csv_path: ../data/data_processing/train_dataset/filtered_dataset.target_frames.clustered.test.csv')
                else:
                    print(line, end='')
        
        # Replace lines in ddpm.py for test set
        #ddpm_replacement = f'        exp_name = \'without_comp_norm_standard_ckpt{step}/test\'\n        DEBUG = True'
        ddpm_replacement = f'        exp_name = \'psearch_a_vis_norm_standard_context{setting}_ckpt{step}/test\'\n        DEBUG = True'
        
        with fileinput.FileInput('../latent_diffusion/ldm/models/diffusion/ddpm.py', inplace=True, backup='.bak') as file:
            for line in file:
                if '#### REPLACEMENT_LINE' in line:
                    print(ddpm_replacement)
                else:
                    print(line, end='')
        
        #### Run with modified config (test set)
        ###try:
        ###    subprocess.run(f'CUDA_VISIBLE_DEVICES=6 python main.py --config {config_file}', shell=True)
        ###except Exception as e:
        ###    print(f"Error in test run: {e}")
        ###    pass
        
        # Restore original files
        restore_files()
        
        print(f"Completed checkpoint: {ckpt}\n")
    
    # Final cleanup
    restore_files()
