This is the autoencoder model. It can be fine-tuned to give sharper output to recorded programs. This is recommended to improve the out of distribution image quality over the pre trained autoencoder.

# How to use computer/autoencoder/config.yaml

This config outlines all the modules, data and trainer arguments used for the autoencoder model. It is read and the settings dictate the instantiation of the target classes. 
- The dataset location is configured in `autoencoder/config.yaml` and should be produced using `data/data_collection/synthetic_script.py` or `data/data_collection/record_script.py` and by processing it using `data/data_processing/main.py` or `data/data_processing/video_convert.py` for multiple or single video/action recording respectively. 
- You can configure the modules, dataset used and trainer settings all in `autoencoder/config.yaml`. The dataset class configured here is from `data/data_processing/datasets.py`. 
- The model supports all auto encoders in `latent_diffusion` which can be downloaded using `latent_diffusion/scripts`. 
- You can configure the pytorch-lightning `trainer` in `config.yaml` with multi-gpu support for training. `num_workers` and `batch_size` are specified in the `data` section.

# How to use computer/autoencoder/sample.py

This script will load a model from a given ckpt `--ckpt_path` and encode and decode images provided to it. You can specify a list of images paths to sample using `--image_paths`. 

1. type `conda activate csllm`
2. type `cd computer`
3. in terminal `python autoencoder/sample.py` 

This will encode and decode some images:

**example command:** `python autoencoder/sample.py --ckpt_path autoencoder/model.ckpt --image_paths pic1.png pic2.png`


# How to use computer/autoencoder/main.py

This function script has some basic functionality for fine-tuning a autoencoder from a ckpt using `--from_ckpt` and with sampling the model afterwards with `sample_model` flag. The samples taken are sourced from the validation datset provided in `autoencoder/config.yaml`. It will train the model as specified in the argument flags and `autoencoder/config.yaml`. The dataset location is configured in `autoencoder/config.yaml` and should be produced using `data/data_collection/synthetic_script.py` and `data/data_processing/main.py`. Note that the dataset needs to be moved to the `computer` directory for training and sampling from it. You can configure the modules, dataset used and trainer settings all in `autoencoder/config.yaml`. After fine-tuning the autoencoder, it can be loaded by the `computer/model` by running `model/main.py` and specifying the path to the new fine-tuned ckpt using `--ae_ckpt`.

1. `conda activate csllm`
2. type `cd computer` (must run from the folder with the processed dataset)
3. in terminal use `python autoencoder/main.py` this will load the model weights as specified in `--from_ckpt` and configuration from `config.yaml` on the dataset specified in the config.

**example command:** `python autoencoder/main.py --save_path autoencoder/train_0 --from_ckpt autoencoder/model_ae.ckpt --sample_model --config autoencoder/config.yaml`

# How to use computer/model/train.py

Simply use `autoencoder/main.py` for training purposes. This just holds the `train_model()` function for initializing the pytorch lightning trainer and calling `trainer.fit()`. Also has an argument that calls `sample_model()` after training for instant model evaluation.