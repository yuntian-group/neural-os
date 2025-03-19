#!/usr/bin/env python3
import yaml
import os
import random
import copy
import argparse
from pathlib import Path
import itertools

def load_yaml(file_path):
    """Load YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, file_path):
    """Save data to YAML file."""
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def generate_configs(base_config_path, output_dir, num_configs=10, random_sampling=True):
    """Generate multiple config variants by modifying key parameters."""
    # Load base config
    base_config = load_yaml(base_config_path)
    base_name = Path(base_config_path).stem
    
    # Define parameter options to explore
    param_options = {
        'temporal_encoder.hidden_size': [4096],
        'unet.attention_resolutions': [ # 8 4 2
            [],
            [2],
            [4],
            [8],
            [2,4],
            [2,8],
            [4,8],
            [2,4,8],
            # [2,4,8,16],
            # [8,2],
            # [4,2,1],
            # [8, 4, 2, 1],
            #[8, 4],
            #[8, 16],
            #[4, 8, 16]
        ],
        'unet.channel_mult': [
            # [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 5],
            #[1,2,4],
            # [1, 2, 3, 4],
            # [1, 2, 3, 5],
            # [1, 2, 4, 4],
            # [1, 1, 2, 2],
            # [1, 2, 3, 4, 5],
            # [1, 2, 3, 5,6],
            # [1, 2, 4, 6, 8],
            # [1, 2, 4, 4, 5],
            # [1, 1, 2, 2, 3],
        ],
        'unet.model_channels':  [320, 384], # [32, 64,128, 160, 192, 224, 256, 320],
        'temporal_encoder.output_channels': [32, 48]
    }
    
    # Create shorthand labels for parameters
    param_labels = {
        'temporal_encoder.hidden_size': 'hs',
        'temporal_encoder.output_channels': 'oc',
        'unet.in_channels': 'nl',
        'unet.attention_resolutions': 'ar',
        'unet.channel_mult': 'cm',
        'unet.model_channels': 'mc',
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    generated_configs = []
    
    if random_sampling:
        # Random sampling approach (select num_configs random combinations)
        for i in range(num_configs):
            config = copy.deepcopy(base_config)
            
            # First select channel_mult
            channel_mult = random.choice(param_options['unet.channel_mult'])
            
            # Then select attention_resolutions that works with this channel_mult
            # Ensure attention_resolutions length <= channel_mult length
            valid_attention_resolutions = [
                ar for ar in param_options['unet.attention_resolutions']
                if len(ar) <= len(channel_mult)
            ]
            
            attention_resolutions = random.choice(valid_attention_resolutions)
            
            # Select other parameters
            hidden_size = random.choice(param_options['temporal_encoder.hidden_size'])
            model_channels = random.choice(param_options['unet.model_channels'])
            output_channels = random.choice(param_options['temporal_encoder.output_channels'])
            
            # Calculate in_channels as 4 + output_channels
            in_channels = 16 + output_channels
            
            # Combine into selected parameters
            selected_params = {
                'temporal_encoder.hidden_size': hidden_size,
                'temporal_encoder.output_channels': output_channels,
                'unet.in_channels': in_channels,
                'unet.attention_resolutions': attention_resolutions,
                'unet.channel_mult': channel_mult,
                'unet.model_channels': model_channels,
            }
            
            # Apply selected parameters to config
            config['model']['params']['temporal_encoder_config']['params']['hidden_size'] = selected_params['temporal_encoder.hidden_size']
            config['model']['params']['temporal_encoder_config']['params']['output_channels'] = selected_params['temporal_encoder.output_channels']
            config['model']['params']['unet_config']['params']['in_channels'] = selected_params['unet.in_channels']
            config['model']['params']['unet_config']['params']['attention_resolutions'] = selected_params['unet.attention_resolutions']
            config['model']['params']['unet_config']['params']['channel_mult'] = selected_params['unet.channel_mult']
            config['model']['params']['unet_config']['params']['model_channels'] = selected_params['unet.model_channels']
            
            # Create a descriptive filename
            config_name = create_descriptive_filename(base_name, selected_params, param_labels)
            
            # Also update the save_path in the config to match the filename
            config['save_path'] = f"saved_{config_name.replace('.yaml', '')}"
            
            # Save config to file
            output_path = os.path.join(output_dir, config_name)
            save_yaml(config, output_path)
            
            # Add to list of generated configs
            generated_configs.append({
                'filename': config_name,
                'params': selected_params
            })
    else:
        # Generate all combinations (warning: can be a large number)
        param_keys = list(param_options.keys())
        
        # Create valid combinations ensuring attention_resolutions length <= channel_mult length
        valid_combinations = []
        for hidden_size in param_options['temporal_encoder.hidden_size']:
            for channel_mult in param_options['unet.channel_mult']:
                for attention_resolutions in param_options['unet.attention_resolutions']:
                    if True:
                        for model_channels in param_options['unet.model_channels']:
                            for output_channels in param_options['temporal_encoder.output_channels']:
                                max_attention_resolution = 2 ** (len(channel_mult) - 1)
                                if len(attention_resolutions) > 0 and max(attention_resolutions) > max_attention_resolution:
                                    continue
                                # Calculate in_channels as 4 + output_channels
                                in_channels = 16 + output_channels
                                valid_combinations.append((
                                    hidden_size,
                                    attention_resolutions,
                                    channel_mult,
                                    model_channels,
                                    output_channels,
                                    in_channels
                                ))
        
        # If too many combinations, sample a subset
        if len(valid_combinations) > num_configs:
            valid_combinations = random.sample(valid_combinations, num_configs)
        valid_combinations.append((1024, [2,4,8], [1,2,3,5], 192, 4, 8))
        
        for i, param_values in enumerate(valid_combinations):
            config = copy.deepcopy(base_config)
            
            # Create a dictionary of selected parameters
            selected_params = {
                'temporal_encoder.hidden_size': param_values[0],
                'unet.attention_resolutions': param_values[1],
                'unet.channel_mult': param_values[2],
                'unet.model_channels': param_values[3],
                'temporal_encoder.output_channels': param_values[4],
                'unet.in_channels': param_values[5],
            }
            
            # Apply selected parameters to config
            config['model']['params']['temporal_encoder_config']['params']['hidden_size'] = selected_params['temporal_encoder.hidden_size']
            config['model']['params']['temporal_encoder_config']['params']['output_channels'] = selected_params['temporal_encoder.output_channels']
            config['model']['params']['unet_config']['params']['in_channels'] = selected_params['unet.in_channels']
            config['model']['params']['unet_config']['params']['attention_resolutions'] = selected_params['unet.attention_resolutions']
            config['model']['params']['unet_config']['params']['channel_mult'] = selected_params['unet.channel_mult']
            config['model']['params']['unet_config']['params']['model_channels'] = selected_params['unet.model_channels']
            
            # Create a descriptive filename
            config_name = create_descriptive_filename(base_name, selected_params, param_labels)
            
            # Also update the save_path in the config to match the filename
            config['save_path'] = f"saved_{config_name.replace('.yaml', '')}"
            
            # Save config to file
            output_path = os.path.join(output_dir, config_name)
            save_yaml(config, output_path)
            
            # Add to list of generated configs
            generated_configs.append({
                'filename': config_name,
                'params': selected_params
            })
    
    # Save summary of generated configs
    summary_file = os.path.join(output_dir, "config_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Generated {len(generated_configs)} configuration variants\n\n")
        for i, config_info in enumerate(generated_configs):
            f.write(f"Config {i+1}: {config_info['filename']}\n")
            for param, value in config_info['params'].items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
    
    print(f"Generated {len(generated_configs)} configuration variants in {output_dir}")
    print(f"Summary saved to {summary_file}")
    
    # Return list of config paths for easy integration with other scripts
    return [os.path.join(output_dir, config_info['filename']) for config_info in generated_configs]

def create_descriptive_filename(base_name, params, param_labels):
    """Create a descriptive filename based on parameters."""
    # Start with the base name
    parts = [base_name]
    
    # Add each parameter's shorthand and value
    for param, label in param_labels.items():
        value = params[param]
        
        if isinstance(value, list):
            # For list parameters, join with underscores
            value_str = '_'.join(map(str, value))
        else:
            value_str = str(value)
        
        parts.append(f"{label}{value_str}")
    
    # Join all parts with underscores and add .yaml extension
    return '_'.join(parts) + '.yaml'

def main():
    parser = argparse.ArgumentParser(description='Generate model configuration variants')
    parser.add_argument('--base-config', default='computer/configs/a.yaml', 
                        help='Path to the base configuration file')
    parser.add_argument('--output-dir', default='computer/configs',
                        help='Directory to save generated configurations')
    parser.add_argument('--num-configs', type=int, default=10000,
                        help='Number of configurations to generate')
    parser.add_argument('--random', action='store_true',
                        help='Use random sampling instead of generating all combinations')
    
    args = parser.parse_args()
    
    # Generate configurations
    config_paths = generate_configs(
        args.base_config, 
        args.output_dir,
        args.num_configs,
        args.random
    )
    
    # Print list of generated config paths
    print("\nGenerated config files:")
    for path in config_paths:
        print(path)
    
    # Print command to run latency measurement
    print("\nTo measure latency across all generated configs, run:")
    configs_str = " ".join(config_paths)
    print(f"python computer/measure_latency.py --configs {configs_str} --frames 50")

if __name__ == "__main__":
    main()
