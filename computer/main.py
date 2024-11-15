from computer.util import load_cond_from_config, load_first_stage_from_config, load_model, load_model_from_config, get_ground_truths, init_model, load_autoencoder_from_ckpt, load_cond_from_ckpt
from computer.train import train_model
from computer.sample import sample_model
from data.data_processing.datasets import DataModule
from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config
import torch
import os
import argparse


#save_path = 'test_15_no_deltas_1000_paths'

##Parse args here

if __name__ == "__main__":

    """
    Trains a model and samples it.
    """

    parser = argparse.ArgumentParser(description='Train and sample a model using a config file')
    parser.add_argument('--config', type=str, default="config_csllm.yaml",
                       help='Path to the configuration file (default: config_csllm.yaml)')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    save_path = config.save_path
    print ('='*10)
    print (save_path)
    print (args.config)
    #print (config.model.scheduler_sampling_rate)
    print ('='*10)
    #import pdb; pdb.set_trace()
    model = init_model(config)
    # model = load_model_from_config(config, "model.ckpt")  # TODO: check path

    # model: LatentDiffusion = load_model_from_config(config, 'model.ckpt')
    model = load_first_stage_from_config(model, './autoencoder_saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_model-603000.ckpt')
    # model = load_cond_from_config(model, "model_bert.ckpt")

    #model = load_model_from_config(config, 'oct29_fixcursor_test_15_no_deltas_1000_paths/model_test_15_no_deltas_1000_paths.ckpt')
    #model = load_model_from_config(config, 'saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_debugidentity_256/model-step=010500.ckpt')
    #model = load_model_from_config(config, 'saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_debugidentity_256_cont/model-step=003500.ckpt')
    #model = load_model_from_config(config, 'oct29_fixcursor_test_15_no_deltas_1000_paths/model_test_15_no_deltas_1000_paths.ckpt')
    #model = load_model_from_config(config, 'saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_debugidentity_256/model_saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_debugidentity_256.ckpt')
    #import pdb; pdb.set_trace()

    #model = load_model_from_config(config, 'test_12_600_epoch_no_deltas/model_test_12_600_epoch_no_deltas.ckpt')
    #model = init_model(config) #initializes the all model modules.

    #model = load_autoencoder_from_ckpt(model, 'autoencoder/train_0/model_ae_epoch=00.ckpt') #loads autoencoder weights.
    #model = load_cond_from_ckpt(model, 'model_bert.ckpt') #loads encoder weights.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    data: DataModule = instantiate_from_config(config.data)
    data.setup()

    print("---------------------------------"); print("\u2705 Model loaded with ae and cond."); print("---------------------------------")

    

    # for name, child in model.model.diffusion_model.named_children():
    #     print(name)
    #     if name == 'input_blocks': print(child)

    #prompts, image_sequences, targets = get_ground_truths(data.datasets['train'], idxs=[i for i in range(173)])

    os.makedirs(save_path, exist_ok=True)

    # sample_model(model, prompts, image_sequences, save_path, create_video=True)

    model = train_model(model, data, save_path, config)
    model = model.to(device)

    #sample_model(model, prompts, image_sequences, save_path, True)
    #prompts, image_sequences, targets = get_ground_truths(data.datasets['train'], idxs=[i for i in range(173)])
