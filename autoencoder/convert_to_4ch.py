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
    
    # Modify encoder.conv_out
    old_encoder_out = state_dict['encoder.conv_out.weight']
    old_encoder_bias = state_dict['encoder.conv_out.bias']
    
    # Create new tensors for encoder output (2*n_channels for mean and std)
    new_encoder_out = torch.zeros((2 * n_channels,) + old_encoder_out.shape[1:])
    new_encoder_bias = torch.zeros(2 * n_channels)
    
    # Transfer weights properly maintaining mean/std pairs
    # Take first n_channels from means (first half)
    new_encoder_out[:n_channels] = old_encoder_out[:n_channels]
    new_encoder_bias[:n_channels] = old_encoder_bias[:n_channels]
    # Take first n_channels from stds (second half)
    new_encoder_out[n_channels:] = old_encoder_out[old_channels:old_channels+n_channels]
    new_encoder_bias[n_channels:] = old_encoder_bias[old_channels:old_channels+n_channels]
    
    state_dict['encoder.conv_out.weight'] = new_encoder_out
    state_dict['encoder.conv_out.bias'] = new_encoder_bias
    
    # Modify quant_conv (encoder bottleneck)
    old_quant_weight = state_dict['quant_conv.weight']
    old_quant_bias = state_dict['quant_conv.bias']
    
    # Create new tensors with proper mean/std pairs and reduced input channels
    new_quant_weight = torch.zeros((2 * n_channels, 2 * n_channels) + old_quant_weight.shape[2:])
    new_quant_bias = torch.zeros(2 * n_channels)
    
    # Transfer weights properly maintaining mean/std pairs
    # Take first n_channels from means (first half)
    new_quant_weight[:n_channels, :n_channels] = old_quant_weight[:n_channels, :n_channels]
    new_quant_bias[:n_channels] = old_quant_bias[:n_channels]
    # Take first n_channels from stds (second half)
    new_quant_weight[n_channels:, n_channels:] = old_quant_weight[old_channels:old_channels+n_channels, old_channels:old_channels+n_channels]
    new_quant_bias[n_channels:] = old_quant_bias[old_channels:old_channels+n_channels]
    
    state_dict['quant_conv.weight'] = new_quant_weight
    state_dict['quant_conv.bias'] = new_quant_bias
    
    # Modify decoder.conv_in
    old_decoder_in = state_dict['decoder.conv_in.weight']
    old_decoder_bias = state_dict['decoder.conv_in.bias']
    
    # Create new tensor with fewer input channels
    new_decoder_in = old_decoder_in[:, :n_channels]
    # Bias remains the same size as it depends on output channels
    
    state_dict['decoder.conv_in.weight'] = new_decoder_in
    state_dict['decoder.conv_in.bias'] = old_decoder_bias
    
    # Modify post_quant_conv (decoder bottleneck)
    old_post_quant_weight = state_dict['post_quant_conv.weight']
    old_post_quant_bias = state_dict['post_quant_conv.bias']
    
    # Create new tensor with reduced input and output channels
    new_post_quant_weight = old_post_quant_weight[:n_channels, :n_channels]
    new_post_quant_bias = old_post_quant_bias[:n_channels]
    
    state_dict['post_quant_conv.weight'] = new_post_quant_weight
    state_dict['post_quant_conv.bias'] = new_post_quant_bias
    
    print(f"Saving converted checkpoint to {output_path}")
    torch.save(checkpoint, output_path)
    print("Done!")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert autoencoder checkpoint to use 4 channels.")
    parser.add_argument("--input", type=str, default='autoencoder_kl_f16.ckpt',
                        help="Path to input checkpoint")
    parser.add_argument("--output", type=str, default='autoencoder_kl_f16_4ch.ckpt',
                        help="Path to save converted checkpoint")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    convert_checkpoint_to_4ch(args.input, args.output)