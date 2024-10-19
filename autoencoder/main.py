from computer.extra import create_prompts
from computer.util import load_cond_from_config, load_first_stage_from_config, load_model, load_model_from_config, get_ground_truths
from computer.autoencoder.train import train_model
from computer.autoencoder.sample import sample_model
from data.data_processing.datasets import DataModule
from latent_diffusion.ldm.models.autoencoder import VQModel
from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config
import torch
import os, argparse
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion

def parse_args():
    parser = argparse.ArgumentParser(description="Runs the finetuning and inference script.")

    parser.add_argument("--save_path", type=str, default='autoencoder/ae_finetune_0',
                        help="where to save the ckpt and resulting samples.")
    
    parser.add_argument("--from_ckpt", type=str, required=True,
                        help="initializes the model from an existing ckpt path.")

    parser.add_argument("--sample_model", action='store_true', default=True,
                        help="Runs some samples on some training data.")

    parser.add_argument("--config", type=str, default="autoencoder/config.yaml",
                        help="specifies the model config to load.")

    return parser.parse_args()



if __name__ == "__main__":

    """
    Fine-tunes the decoder side of a autoencoder model and samples it.
    """

    args=parse_args()

    config = OmegaConf.load(args.config)

    torch.cuda.empty_cache()
    model: VQModel = load_model_from_config(config, args.from_ckpt)

    data: DataModule = instantiate_from_config(config.data)
    data.setup()

    print("---------------------------------"); print("\u2705 Model loaded..."); print("---------------------------------")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    _, _, targets = get_ground_truths(data.datasets['validation'], idxs=[i for i in range(173)])

    model, rank = train_model(model, data, args.save_path, config)

    if args.sample_model and rank:
        model = model.to(device)
        model.eval()
        model = torch.compile(model, mode='max-autotune', fullgraph=True)
        sample_model(model, targets, args.save_path + "/sample", True)
