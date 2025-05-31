#!/bin/bash
#SBATCH --account=def-yuntian
#SBATCH --job-name=train_ds
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --gres=gpu:h100:8
#SBATCH --time=7-00:00:00

source ~/.bashrc

cd /home/yuntian/projects/def-yuntian/yuntian/computer/computer
# Important: Set CUDA memory allocator explicitly
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# If your PyTorch is >= 2.0, you can add (but it's optional, remove if unsupported)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

nvidia-smi

python main.py --config configs/fb_computecanada_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial.yaml > log.fb.computecanada.challengingandsample.pretrainrnn.balanced.lr5e6.contbest.samplercover.newd.contfreezernn.newnewd.origunet.nospatial2 2>&1
