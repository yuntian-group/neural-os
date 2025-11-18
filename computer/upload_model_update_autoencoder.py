import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, create_repo
from safetensors.torch import save_file
import os
from omegaconf import OmegaConf
import json
import glob

# Screen resolution constants
OLD_SCREEN_WIDTH = 512
OLD_SCREEN_HEIGHT = 384
NEW_SCREEN_WIDTH = 1024
NEW_SCREEN_HEIGHT = 768

# Model details
MODEL_NAME = "yuntian-deng/computer-doom-nibik-032k-updated-autoencoder-16x"

# Paths
DIFFUSION_MODEL_CHECKPOINT = "./test_scheduled_samplingk2777doom222222_pretrainrnn22_nibik/model-step=032000.ckpt"
AUTOENCODER_CHECKPOINT_DIR = "../autoencoder/saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384_mar10_keyboard_1024_768_4.5e7_cont_1e5_forui_real_4.5e6_4.5e7_gan_p1.0"
AUTOENCODER_CONFIG = "../autoencoder/config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard_1024_768.4.5e7.cont.1e5.forui.real.4.5e7.gan.p1.0.yaml"
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

def resize_cursor_position_embeddings(state_dict, old_screen_width, old_screen_height, new_screen_width, new_screen_height, latent_width, latent_height):
    """
    Resize cursor position embeddings (embedding_x, embedding_y) to match new screen resolution.
    Note: image_position_embeddings does NOT change since latent size remains the same.
    
    Args:
        state_dict: Model state dict
        old_screen_width, old_screen_height: Old screen resolution (512x384)
        new_screen_width, new_screen_height: New screen resolution (1024x768)
        latent_width, latent_height: Latent dimensions (should be same, e.g., 64x48)
    """
    # Calculate embedding sizes based on screen resolution
    # Old: embedding_x size = latent_width * 8 = 64 * 8 = 512 (matches old screen width)
    # New: embedding_x size = latent_width * 16 = 64 * 16 = 1024 (matches new screen width)
    old_x_size = old_screen_width  # Should be 512
    new_x_size = new_screen_width  # Should be 1024
    x_key = 'temporal_encoder.embedding_x.weight'
    if x_key in state_dict:
        old_x_emb = state_dict[x_key]
        if old_x_emb.shape[0] != new_x_size:
            print(f"Resizing {x_key}: {old_x_emb.shape[0]} -> {new_x_size} (screen width: {old_screen_width} -> {new_screen_width})")
            # Interpolate embedding weights
            old_x_emb_2d = old_x_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, old_size, hidden_size]
            new_x_emb_2d = F.interpolate(
                old_x_emb_2d.transpose(1, 3),  # [1, hidden_size, old_size, 1]
                size=(new_x_size, 1),
                mode='bilinear',
                align_corners=False
            ).transpose(1, 3).squeeze(0).squeeze(0)  # [new_size, hidden_size]
            state_dict[x_key] = new_x_emb_2d
        else:
            print(f"{x_key} size unchanged: {old_x_emb.shape[0]}")
    
    # Resize embedding_y
    old_y_size = old_screen_height  # Should be 384
    new_y_size = new_screen_height  # Should be 768
    y_key = 'temporal_encoder.embedding_y.weight'
    if y_key in state_dict:
        old_y_emb = state_dict[y_key]
        if old_y_emb.shape[0] != new_y_size:
            print(f"Resizing {y_key}: {old_y_emb.shape[0]} -> {new_y_size} (screen height: {old_screen_height} -> {new_screen_height})")
            # Interpolate embedding weights
            old_y_emb_2d = old_y_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, old_size, hidden_size]
            new_y_emb_2d = F.interpolate(
                old_y_emb_2d.transpose(1, 3),  # [1, hidden_size, old_size, 1]
                size=(new_y_size, 1),
                mode='bilinear',
                align_corners=False
            ).transpose(1, 3).squeeze(0).squeeze(0)  # [new_size, hidden_size]
            state_dict[y_key] = new_y_emb_2d
        else:
            print(f"{y_key} size unchanged: {old_y_emb.shape[0]}")
    
    # Verify image_position_embeddings is NOT changed (latent size unchanged)
    img_pos_key = 'temporal_encoder.image_position_embeddings'
    if img_pos_key in state_dict:
        img_pos = state_dict[img_pos_key]
        expected_hw = latent_height * latent_width
        if img_pos.shape[1] == expected_hw:
            print(f"{img_pos_key} unchanged (latent size {latent_height}x{latent_width} = {expected_hw} positions)")
        else:
            print(f"WARNING: {img_pos_key} has unexpected size {img_pos.shape[1]}, expected {expected_hw}")

def update_autoencoder_in_state_dict(diffusion_checkpoint_path, autoencoder_checkpoint_path, diffusion_config_path):
    """
    Update the first_stage_model (autoencoder) in a diffusion model state_dict and resize cursor position embeddings.
    
    Args:
        diffusion_checkpoint_path: Path to the diffusion model checkpoint
        autoencoder_checkpoint_path: Path to the new autoencoder checkpoint
        diffusion_config_path: Path to diffusion config (to get latent dimensions)
    
    Returns:
        Updated state_dict ready for saving
    """
    print(f"Loading diffusion model checkpoint from: {diffusion_checkpoint_path}")
    diffusion_ckpt = torch.load(diffusion_checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"Loading autoencoder checkpoint from: {autoencoder_checkpoint_path}")
    autoencoder_ckpt = torch.load(autoencoder_checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load diffusion config to get latent dimensions
    diffusion_config = OmegaConf.load(diffusion_config_path)
    
    # Get latent dimensions from diffusion config (latent size doesn't change)
    latent_height = diffusion_config.model.params.temporal_encoder_config.params.output_height
    latent_width = diffusion_config.model.params.temporal_encoder_config.params.output_width
    print(f"Latent dimensions: {latent_height}x{latent_width} (unchanged)")
    
    # Screen resolutions: old was 512x384, new is 1024x768
    # These correspond to the change from 8x to 16x downsampling
    print(f"Screen resolution change: {OLD_SCREEN_WIDTH}x{OLD_SCREEN_HEIGHT} -> {NEW_SCREEN_WIDTH}x{NEW_SCREEN_HEIGHT}")
    
    # Get state dicts
    diffusion_state_dict = diffusion_ckpt['state_dict'].copy()  # Make a copy to avoid modifying original
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
    
    # Debug: Show some example keys being added
    print(f"\nExample autoencoder keys (first 5):")
    for i, orig_key in enumerate(list(autoencoder_state_dict.keys())[:5]):
        new_key = orig_key.replace('model.', '', 1) if orig_key.startswith('model.') else orig_key
        print(f"  {orig_key} -> first_stage_model.{new_key}")
    
    # Add new keys to diffusion state dict
    diffusion_state_dict.update(autoencoder_keys_to_add)
    print(f"\nAdded {len(autoencoder_keys_to_add)} new first_stage_model keys from autoencoder")
    
    # Resize cursor position embeddings based on screen resolution change
    # Note: image_position_embeddings does NOT change since latent size is unchanged
    print(f"\nResizing cursor position embeddings for screen resolution change")
    resize_cursor_position_embeddings(
        diffusion_state_dict,
        OLD_SCREEN_WIDTH, OLD_SCREEN_HEIGHT,
        NEW_SCREEN_WIDTH, NEW_SCREEN_HEIGHT,
        latent_width, latent_height  # Latent dimensions (unchanged)
    )
    
    # Verify some keys exist
    sample_keys = ['first_stage_model.encoder.conv_in.weight', 'first_stage_model.decoder.conv_out.weight']
    print(f"\nVerifying keys exist in updated state_dict:")
    for key in sample_keys:
        if key in diffusion_state_dict:
            print(f"  ✓ {key} (shape: {diffusion_state_dict[key].shape})")
        else:
            print(f"  ✗ {key} NOT FOUND")
    
    return diffusion_state_dict

def upload_model_to_hub():
    # Find latest autoencoder checkpoint
    autoencoder_ckpt_path = find_latest_autoencoder_checkpoint(AUTOENCODER_CHECKPOINT_DIR)
    
    # Update the state_dict with new autoencoder (no intermediate file needed)
    state_dict = update_autoencoder_in_state_dict(
        DIFFUSION_MODEL_CHECKPOINT,
        autoencoder_ckpt_path,
        DIFFUSION_CONFIG
    )
    
    # Load the configuration
    config = OmegaConf.load(DIFFUSION_CONFIG)
    
    # Create or get the repo
    api = HfApi()
    repo_url = create_repo(MODEL_NAME, private=False)
    
    # Save the state_dict to safetensors format
    save_file(state_dict, "model.safetensors")
    
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

if __name__ == "__main__":
    upload_model_to_hub()

