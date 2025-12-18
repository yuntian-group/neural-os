import torch
import os

d = 'saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384_mar10_keyboard_1024_768_4.5e7_cont_1e5_forui_real_4.5e6_4.5e7_gan_p1.0_newdata_finalrun_balanced_acc32_reload'

a = torch.load(os.path.join(d, 'model-step=098500.ckpt'))
del a['loops']
#del a['optimizer_states']

# Fix logvar shape: convert from scalar to 1D tensor [1] for FSDP compatibility
if 'state_dict' in a and 'loss.logvar' in a['state_dict']:
    logvar_value = a['state_dict']['loss.logvar']
    if logvar_value.dim() == 0:  # scalar tensor
        logvar_init = logvar_value.item()
        device = logvar_value.device
        a['state_dict']['loss.logvar'] = torch.tensor([logvar_init], device=device)
        print(f"Fixed logvar shape: converted scalar {logvar_init} to tensor shape [1]")

# Remove loss.logvar from optimizer state dict (FSDP can't map it if it's not in current optimizer)
# FSDP uses parameter names as keys, so we look for any key containing 'loss.logvar'
if 'optimizer_states' in a:
    for opt_idx, opt_state in enumerate(a['optimizer_states']):
        removed_count = 0
        # Remove from state dict (FSDP uses parameter names as keys)
        if 'state' in opt_state:
            keys_to_remove = [k for k in opt_state['state'].keys() if 'loss.logvar' in str(k)]
            for key in keys_to_remove:
                del opt_state['state'][key]
                removed_count += 1
        
        if removed_count > 0:
            print(f"Removed {removed_count} loss.logvar entries from optimizer {opt_idx} state")

torch.save(a, os.path.join(d, 'model-step=098500.noloops.ckpt'))
