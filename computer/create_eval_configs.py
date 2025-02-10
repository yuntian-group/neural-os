import os
import shutil
import yaml

config_dir = 'configs'
config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml') and 'context' in f and 'eval' not in f]

for config_file in config_files:
    # Create eval filename
    eval_file = config_file.replace('.yaml', '.eval.yaml')
    
    # Read original file
    with open(os.path.join(config_dir, config_file), 'r') as f:
        config = yaml.safe_load(f)
    
    # Make modifications
    # Update save path
    if 'save_path' in config:
        config['save_path'] = config['save_path'] + '_eval'
    
    # Update accumulate_grad_batches
    config['lightning']['trainer']['accumulate_grad_batches'] = 99999999
        
    # Update data csv path
    config['data']['params']['train']['params']['data_csv_path'] = config['data']['params']['train']['params']['data_csv_path'].replace('.train.', '.test.')
    
    # Write modified config to eval file
    with open(os.path.join(config_dir, eval_file), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
print("Created eval configs for:")
for f in config_files:
    print(f"- {f} -> {f.replace('.yaml', '.eval.yaml')}")
