import torch
from huggingface_hub import HfApi, create_repo
import os
from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config
from computer.util import load_model_from_config
import json
import glob

# Model details
MODEL_NAME = "yuntian-deng/computer-doom-nibik-032k-updated-autoencoder-16x"

# Paths
DIFFUSION_MODEL_CHECKPOINT = "./test_scheduled_samplingk2777doom222222_pretrainrnn22_nibik/model-step=032000.ckpt"
AUTOENCODER_CHECKPOINT_DIR = "../autoencoder/saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384_mar10_keyboard_1024_768_4.5e7_cont_1e5_forui_real_4.5e6_4.5e7_gan_p1.0"
AUTOENCODER_CONFIG = "autoencoder/config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard_1024_768.4.5e7.cont.1e5.forui.real.4.5e7.gan.p1.0.yaml"
DIFFUSION_CONFIG = "config_csllm.yaml"  # Update this to your diffusion model config

def find_latest_autoencoder_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the autoencoder directory."""
    # Look for checkpoint files
    checkpoint_patterns = [
        os.path.join(checkpoint_dir, "model-step=*.ckpt"),
        os.path.join(checkpoint_dir, "model-*.ckpt"),
        os.path.join(checkpoint_dir, "*.ckpt"),
    ]
    
    all_checkpoints = []
    for pattern in checkpoint_patterns:
        all_checkpoints.extend(glob.glob(pattern))
    
    if not all_checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Sort by modification time (latest first)
    all_checkpoints.sort(key=os.path.getmtime, reverse=True)
    latest_checkpoint = all_checkpoints[0]
    
    print(f"Found latest autoencoder checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def update_autoencoder_in_checkpoint(diffusion_checkpoint_path, autoencoder_checkpoint_path, output_path=None):
    """
    Update the first_stage_model (autoencoder) in a diffusion model checkpoint.
    
    Args:
        diffusion_checkpoint_path: Path to the diffusion model checkpoint
        autoencoder_checkpoint_path: Path to the new autoencoder checkpoint
        output_path: Optional path to save the updated checkpoint (if None, overwrites original)
    """
    print(f"Loading diffusion model checkpoint from: {diffusion_checkpoint_path}")
    diffusion_ckpt = torch.load(diffusion_checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"Loading autoencoder checkpoint from: {autoencoder_checkpoint_path}")
    autoencoder_ckpt = torch.load(autoencoder_checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get state dicts
    diffusion_state_dict = diffusion_ckpt['state_dict']
    autoencoder_state_dict = autoencoder_ckpt['state_dict']
    
    # Find all first_stage_model keys in diffusion checkpoint
    first_stage_keys = [k for k in diffusion_state_dict.keys() if k.startswith('first_stage_model.')]
    print(f"Found {len(first_stage_keys)} first_stage_model keys in diffusion checkpoint")
    
    # Remove old first_stage_model keys
    for key in first_stage_keys:
        del diffusion_state_dict[key]
    
    print(f"Removed old first_stage_model keys")
    
    # Add new first_stage_model keys from autoencoder checkpoint
    # The autoencoder checkpoint keys might have 'model.' prefix or be direct
    autoencoder_keys_to_add = {}
    for key, value in autoencoder_state_dict.items():
        # Remove 'model.' prefix if present (Lightning sometimes adds this)
        new_key = key.replace('model.', '', 1) if key.startswith('model.') else key
        # Add 'first_stage_model.' prefix to match diffusion model structure
        first_stage_key = f'first_stage_model.{new_key}'
        autoencoder_keys_to_add[first_stage_key] = value
    
    # Add new keys to diffusion state dict
    diffusion_state_dict.update(autoencoder_keys_to_add)
    print(f"Added {len(autoencoder_keys_to_add)} new first_stage_model keys from autoencoder")
    
    # Update checkpoint
    diffusion_ckpt['state_dict'] = diffusion_state_dict
    
    # Save updated checkpoint
    if output_path is None:
        output_path = diffusion_checkpoint_path.replace('.ckpt', '_updated_autoencoder.ckpt')
    
    print(f"Saving updated checkpoint to: {output_path}")
    torch.save(diffusion_ckpt, output_path)
    
    return output_path

def upload_model_to_hub():
    # Find latest autoencoder checkpoint
    autoencoder_ckpt_path = find_latest_autoencoder_checkpoint(AUTOENCODER_CHECKPOINT_DIR)
    
    # Update the checkpoint with new autoencoder
    updated_checkpoint_path = update_autoencoder_in_checkpoint(
        DIFFUSION_MODEL_CHECKPOINT,
        autoencoder_ckpt_path
    )
    
    # Load the configuration
    config = OmegaConf.load(DIFFUSION_CONFIG)
    
    # Load the updated checkpoint
    print(f"Loading updated checkpoint: {updated_checkpoint_path}")
    state_dict = torch.load(updated_checkpoint_path, map_location='cpu', weights_only=False)['state_dict']
    
    # Create or get the repo
    api = HfApi()
    repo_url = create_repo(MODEL_NAME, private=False)
    
    # Save the state_dict to a file
    torch.save(state_dict, "model.safetensors")
    
    # Convert OmegaConf to a regular dict and save as JSON
    config_dict = OmegaConf.to_container(config, resolve=True)
    with open("config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Create a README file
    with open("README.md", "w") as f:
        f.write(f"# {MODEL_NAME}\n\n")
        f.write("This is a LatentDiffusion model for frame prediction.\n\n")
        f.write(f"**Updated autoencoder**: Uses 16x downsampling factor (updated from 8x)\n\n")
        f.write(f"**Original checkpoint**: {DIFFUSION_MODEL_CHECKPOINT}\n\n")
        f.write(f"**Autoencoder checkpoint**: {autoencoder_ckpt_path}\n")
    
    # Push the files to the hub
    api.upload_file(
        path_or_fileobj="model.safetensors",
        path_in_repo="model.safetensors",
        repo_id=MODEL_NAME,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="config.json",
        path_in_repo="config.json",
        repo_id=MODEL_NAME,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=MODEL_NAME,
        repo_type="model",
    )
    
    print(f"Model uploaded successfully to: https://huggingface.co/{MODEL_NAME}")
    
    # Clean up the temporary files
    os.remove("model.safetensors")
    os.remove("config.json")
    os.remove("README.md")
    
    print(f"Updated checkpoint saved to: {updated_checkpoint_path}")

if __name__ == "__main__":
    upload_model_to_hub()

