import torch
import sys

def keep_only_state_dict(ckpt_path):
    output_path = ckpt_path + '.state_dict_only'
    # Load checkpoint
    #ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)['state_dict']
    
    # Extract state_dict
    #state_dict = ckpt['state_dict']
    
    # Save the simplified checkpoint
    torch.save({'state_dict': state_dict}, output_path)

# Example usage:
keep_only_state_dict(sys.argv[1])

