import torch
from huggingface_hub import HfApi, create_repo
import os
from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config
import json

# Set your Hugging Face token here
# Set your model details
MODEL_NAME = "yuntian-deng/computer-model"
LOCAL_CHECKPOINT_PATH = "test_15_no_deltas_1000_paths/model_test_15_no_deltas_1000_paths.ckpt"
LOCAL_CHECKPOINT_PATH = "oct27_test_15_no_deltas_1000_paths/model_test_15_no_deltas_1000_paths.ckpt"
LOCAL_CHECKPOINT_PATH = "checkpoints/model-step=007500.ckpt"
LOCAL_CHECKPOINT_PATH = "test_15_no_deltas_1000_paths/model_test_15_no_deltas_1000_paths.ckpt"
LOCAL_CHECKPOINT_PATH = "test_15_no_deltas_1000_paths/model_test_15_no_deltas_1000_paths.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5/model_saved_fixcursor_lr2e5.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug/model-step=004000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe/model-step=002000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe/model-step=005000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_identity/model_saved_fixcursor_lr2e5_debug_gpt_firstframe_identity.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap/model-step=001000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap/model-step=001000-v1.ckpt"
#LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap/model-step=005000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap/model_saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_debugidentity/model-step=009000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_identity/model-step=005000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_debugidentity_256_cont/model-step=003000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_longtrainh200/model-step=007500.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_bsz64_acc1_lr8e5/model-step=012500.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_bsz64_acc1_lr8e5_512_leftclick/model-step=246000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_bsz64_acc1_lr8e5_512_leftclick_histpos/model-step=043000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2/model-step=016000.ckpt"
CONFIG_PATH = "config_csllm.yaml"
CONFIG_PATH = "configs/2e5_debug_gpt_firstframe.yaml"
CONFIG_PATH = "configs/2e5_debug_gpt_firstframe_identity.yaml"
CONFIG_PATH = "configs/2e5_debug_gpt_firstframe_posmap_debugidentity.yaml"
CONFIG_PATH = "configs/2e5_debug_gpt_firstframe_posmap_longtrainh200.yaml"
CONFIG_PATH = "configs/pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos.yaml"
CONFIG_PATH = "configs/pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384.yaml"

def upload_model_to_hub():
    # Load the configuration
    config = OmegaConf.load(CONFIG_PATH)
    
    # Load the local checkpoint
    checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location='cpu')
    
    # Extract only the state_dict
    state_dict = checkpoint['state_dict']
    
    # Create or get the repo
    api = HfApi()
    #repo_url = create_repo(MODEL_NAME, private=True)
    
    # Save the state_dict to a file
    torch.save(state_dict, "model.safetensors")
    
    # Convert OmegaConf to a regular dict and save as JSON
    config_dict = OmegaConf.to_container(config, resolve=True)
    with open("config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Create a README file
    with open("README.md", "w") as f:
        f.write(f"# {MODEL_NAME}\n\nThis is a LatentDiffusion model for frame prediction.")
    
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
