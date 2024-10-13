from computer.util import load_cond_from_config, load_first_stage_from_config, load_model, load_model_from_config, get_ground_truths
from computer.train import train_model
from computer.sample import sample_model
from data.data_processing.datasets import DataModule
from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config
import torch
import os


save_path = 'test_15_no_deltas_1000_paths'

##Parse args here

if __name__ == "__main__":

    """
    Trains a model and samples it.
    """

    config = OmegaConf.load("config_csllm.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    # model = load_model_from_config(config, "model.ckpt")  # TODO: check path

    # model: LatentDiffusion = load_model_from_config(config, 'model.ckpt')
    # model = load_first_stage_from_config(model, "model_ae.ckpt")
    # model = load_cond_from_config(model, "model_bert.ckpt")

    # model = load_model_from_config(config, 'test_12_600_epoch_no_deltas/model_test_12_600_epoch_no_deltas.ckpt')

    data: DataModule = instantiate_from_config(config.data)
    data.setup()

    print("---------------------------------"); print("\u2705 Model loaded with ae and cond."); print("---------------------------------")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # for name, child in model.model.diffusion_model.named_children():
    #     print(name)
    #     if name == 'input_blocks': print(child)

    prompts, image_sequences, targets = get_ground_truths(data.datasets['train'], idxs=[i for i in range(173)])

    os.makedirs(save_path, exist_ok=True)

    # sample_model(model, prompts, image_sequences, save_path, create_video=True)

    model = train_model(model, data, save_path, config)
    model = model.to(device)

    sample_model(model, prompts, image_sequences, save_path, True)
    prompts, image_sequences, targets = get_ground_truths(data.datasets['train'], idxs=[i for i in range(173)])