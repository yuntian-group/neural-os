import torch
import os

d = 'saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384_mar10_keyboard_1024_768_4.5e7_cont_1e5_forui_real_4.5e6_4.5e7_gan_p1.0_newdata_finalrun_balanced_acc32'

a = torch.load(os.path.join(d, 'model-step=001750.ckpt'))
del a['loops']
#del a['optimizer_states']
torch.save(a, os.path.join(d, 'model-step=001750.noloops.ckpt'))
