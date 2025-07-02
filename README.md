# NeuralOS: Towards Simulating Operating Systems via Neural Generative Models

This repository contains the code for reproducing **NeuralOS**, a neural framework that simulates graphical user interfaces (GUIs) of operating systems by directly predicting screen frames from user inputs.

## Online Demo

Try our demo at [https://neural-os.com/](https://neural-os.com/)!

## Abstract

We introduce NeuralOS, a neural framework that simulates graphical user interfaces (GUIs) of operating systems by directly predicting screen frames from user inputs such as mouse movements, clicks, and keyboard events. NeuralOS combines a recurrent neural network (RNN), which tracks computer state, with a diffusion-based neural renderer that generates desktop images. The model is trained on a large-scale dataset of Ubuntu XFCE recordings, which include both randomly generated interactions and realistic interactions produced by AI agents. Our experiments demonstrate that NeuralOS successfully renders realistic GUI sequences, accurately captures mouse interactions, and reliably predicts state transitions like application launches. Although precisely modeling fine-grained keyboard interactions remains challenging, NeuralOS offers a step toward creating fully adaptive, generative neural interfaces for future human-computer interaction systems.

## Base Framework

This repository is based on the [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion/tree/main) framework by CompVis.

## Repository Structure

- **`data/`**: Code for generating and processing training data
- **`autoencoder/`**: Code for training the autoencoder to reduce image resolution
- **`computer/`**: Code for training the main NeuralOS model

## Quick Start

### 1. Data Generation

First, build the Docker image for our data collection environment:

```
cd data/data_collection
docker build -t synthetic_data_generator .
```

This builds a Docker image with a lower resolution (512×384) to make training compute-feasible.

After building the Docker image, collect training data:

```
python synthetic_script.py
```

This uses up to 64 workers to collect 40K random interactions with the OS. Each interaction is 30 seconds long with actions issued 15 times per second (15fps recording rate). The recorded videos and randomly generated interactions are saved in the `raw_data/` folder.

### 2. Data Processing

After data collection, aggregate all collected videos and actions:

```
cd data/data_processing
python main.py
```

This creates a CSV file (for record and frame numbers) and a PKL file (for mouse and keyboard actions corresponding to each record and frame).

### 3. Autoencoder Training

Train the autoencoder to reduce image resolution by 8x (from 512×384×3 to 64×48×16 latent images):

```
cd autoencoder/
python main.py --config config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard_init_16_contmar15_acc1_cont1e6.yaml
```

**Note**: Training the autoencoder takes approximately 1 week on 1 H200 GPU for about 1 million gradient steps to achieve good results.

### 4. Image Preprocessing

Process raw images using the trained autoencoder to convert them to latent images:

```
cd autoencoder/
python preprocess_dataset.py
```

**Important**: Set `ckpt_path` to point to your trained autoencoder and configure data paths correctly.

This saves all processed latent images in WebDataset tar files (one per video) for training.

### 5. Main Model Training

Launch training of the main NeuralOS model:

```
cd computer/
python main.py --config configs/fb_computecanada_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial_online_x0_joint_onlineonly_7.yaml
```

**Configuration Notes**:
- Modify data paths in the config file to point to your data files
- Training uses 8 GPUs on a single node with FSDP
- Training occurs in multiple stages controlled by config flags:
  - `"pretrain": true` → RNN pretraining (see paper)
  - `"freezernn": true` → Freeze RNN, train only diffusion UNet (`"pretrain"` must be `false`)
  - `"data/params/train/context_length"` → Controls context length
  - `"data/params/train/data_csv_paths"` → Training file paths (supports multiple files)

**Hardware Requirements**:
- Batch size is configured for 8×80GB H100 GPUs
- Training used 8×140GB H200 GPUs
- Adjust batch size based on available GPU memory

### 6. Model Upload

Upload your trained model to Hugging Face:

```
python upload_model.py
```

**Important**: Set model names and checkpoint paths correctly before running.

## Acknowledgments

This research was supported by Compute Canada through the Resources for Research Groups (RRG) 2025 competition, awarded to Yuntian Deng (RRG No. 5275), and was also partially supported by collaborative research funding from the National Research Council of Canada's Artificial Intelligence for Design Program (AI4D-150). Additionally, Yuntian Deng acknowledges support from an NSERC Discovery Grant (RGPIN-2024-05178) and a Starter Grant provided by the University of Waterloo.

## Citation

If you use this code in your research, please cite our paper:

```
@article{neuralOS2025,
  title={NeuralOS: Towards Simulating Operating Systems via Neural Generative Models},
  author={Luke Rivard and Sun Sun and Hongyu Guo and Wenhu Chen and Yuntian Deng},
  year={2025}
}
```
