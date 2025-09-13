#!/bin/bash
#SBATCH --account=aip-yuntian
#SBATCH --job-name=distill_64_to_32
#SBATCH --output=logs/distill_64_to_32_%j.out
#SBATCH --error=logs/distill_64_to_32_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --gres=gpu:h100:8
#SBATCH --time=3-00:00:00

source ~/.bashrc

cd /home/yuntian/projects/aip-yuntian/yuntian/computer/computer

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# Reasonable defaults for multi-GPU stability
export OMP_NUM_THREADS=6
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_BLOCKING_WAIT=0

CUDA_VISIBLE_DEVICES=1 python -u main.py --config configs/train_progressive_distillation_64_to_32.yaml > log.train_progressive_distillation_64_to_32.bsz63.txt.freezernn 2>&1
CUDA_VISIBLE_DEVICES=1 python -u main.py --config configs/train_progressive_distillation_32_to_16.yaml > log.train_progressive_distillation_32_to_16.bsz63.txt.freezernn 2>&1
CUDA_VISIBLE_DEVICES=1 python -u main.py --config configs/train_progressive_distillation_16_to_8.yaml > log.train_progressive_distillation_16_to_8.bsz63.txt.freezernn 2>&1
CUDA_VISIBLE_DEVICES=1 python -u main.py --config configs/train_progressive_distillation_8_to_4.yaml > log.train_progressive_distillation_8_to_4.bsz63.txt.freezernn 2>&1


CUDA_VISIBLE_DEVICES=1 python -u main.py --config configs/train_progressive_distillation_32_to_16_32.yaml > log.train_progressive_distillation_32_to_16_32.bsz63.txt.freezernn 2>&1
