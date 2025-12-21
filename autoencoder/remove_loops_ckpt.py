import torch
import os

d = 'saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384_mar10_keyboard_1024_768_4.5e7_cont_1e5_forui_real_4.5e6_4.5e7_gan_p1.0_newdata_finalrun_balanced_acc32_reload_4gpus_cont_4.5e6_1e6_5e7'

a = torch.load(os.path.join(d, 'model-step=051000.ckpt'))
del a['loops']
del a['optimizer_states']

# Fix logvar shape: convert from scalar to 1D tensor [1] for FSDP compatibility
if 'state_dict' in a and 'loss.logvar' in a['state_dict']:
    logvar_value = a['state_dict']['loss.logvar']
    if logvar_value.dim() == 0:  # scalar tensor
        logvar_init = logvar_value.item()
        device = logvar_value.device
        a['state_dict']['loss.logvar'] = torch.tensor([logvar_init], device=device)
        print(f"Fixed logvar shape: converted scalar {logvar_init} to tensor shape [1]")


torch.save(a, os.path.join(d, 'model-step=051000.noloops.ckpt'))
