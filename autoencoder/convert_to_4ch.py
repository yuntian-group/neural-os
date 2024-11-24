import torch
import argparse

def convert_checkpoint_to_4ch(input_path, output_path):
    """Convert a checkpoint to use 4 channels in the latent space"""
    print(f"Loading checkpoint from {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Get the original number of channels
    old_channels = state_dict['quant_conv.weight'].shape[0] // 2  # divide by 2 because of mean and std
    n_channels = 4  # target number of channels
    print(f"Converting from {old_channels} to {n_channels} channels")
    
    # Modify quant_conv (encoder bottleneck)
    old_quant_weight = state_dict['quant_conv.weight']
    old_quant_bias = state_dict['quant_conv.bias']
    
    # Create new tensors with proper mean/std pairs
    new_quant_weight = torch.zeros((2 * n_channels,) + old_quant_weight.shape[1:])
    new_quant_bias = torch.zeros(2 * n_channels)
    
    # Transfer weights properly maintaining mean/std pairs
    # Take first n_channels from means (first half)
    new_quant_weight[:n_channels] = old_quant_weight[:n_channels]
    new_quant_bias[:n_channels] = old_quant_bias[:n_channels]
    # Take first n_channels from stds (second half)
    new_quant_weight[n_channels:] = old_quant_weight[old_channels:old_channels+n_channels]
    new_quant_bias[n_channels:] = old_quant_bias[old_channels:old_channels+n_channels]
    
    state_dict['quant_conv.weight'] = new_quant_weight
    state_dict['quant_conv.bias'] = new_quant_bias
    
    # Modify post_quant_conv (decoder bottleneck)
    old_post_quant_weight = state_dict['post_quant_conv.weight']
    old_post_quant_bias = state_dict['post_quant_conv.bias']
    
    # Create new tensor with fewer input channels
    new_post_quant_weight = old_post_quant_weight[:, :n_channels]
    # Bias remains the same size as it depends on output channels
    
    state_dict['post_quant_conv.weight'] = new_post_quant_weight
    state_dict['post_quant_conv.bias'] = old_post_quant_bias
    
    print(f"Saving converted checkpoint to {output_path}")
    torch.save(checkpoint, output_path)
    print("Done!")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert autoencoder checkpoint to use 4 channels.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save converted checkpoint")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    convert_checkpoint_to_4ch(args.input, args.output) 