from computer.util import init_model, get_ground_truths
from train import train_model
#from sample import sample_model_from_dataset
from data.data_processing.datasets import DataModule
from latent_diffusion.ldm.models.autoencoder import VQModel
from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config
import torch
import os, argparse
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion

def parse_args():
    parser = argparse.ArgumentParser(description="Runs the finetuning and inference script.")

    #parser.add_argument("--save_path", type=str, default=os.path.join('autoencoder_vq8', 'ae_finetune_0'),
    #                    help="where to save the ckpt and resulting samples.")
    
    parser.add_argument("--from_ckpt", type=str, default='kl-f8.ckpt',
                        help="initializes the model from an existing ckpt path.")

    parser.add_argument("--sample_model", action='store_true', default=True,
                        help="Runs some samples on some training data.")

    parser.add_argument("--config", type=str, default=os.path.join("config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard.yaml"),
                        help="specifies the model config to load.")

    return parser.parse_args()



if __name__ == "__main__":

    """
    Fine-tunes the decoder side of a autoencoder model and samples it.
    """

    args=parse_args()

    config = OmegaConf.load(args.config)
    save_path = config.save_path

    torch.cuda.empty_cache()
    model: VQModel = init_model(config) #initializes the model modules.

    data: DataModule = instantiate_from_config(config.data) # python autoencoder/main.py --from_ckpt autoencoder/512_test/model_ae_epoch=1.ckpt --save_path autoencoder/512_test --config autoencoder/config_vq8.yaml

    print("---------------------------------"); print("\u2705 Model loaded..."); print("---------------------------------")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, rank = train_model(model, data, save_path, config, ckpt_path=args.from_ckpt)

    #if args.sample_model and rank:
    #    model = model.to(device)
    #    model.eval()
    #    # model = torch.compile(model, mode='max-autotune', fullgraph=True)
    #    sample_model_from_dataset(
    #        model, 
    #        data.datasets['train'], 
    #        idxs=[i for i in range(150)], 
    #        save_path=os.path.join(args.save_path, "sample"), 
    #        create_video=True,
    #        targets=True
    #    )
