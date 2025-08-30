#!/bin/bash
#SBATCH --account=aip-yuntian
#SBATCH --job-name=train_ds
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
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

srun python main.py --config configs/test_scheduled_sampling.4steps.yaml > log.test_scheduled_sampling.4steps 2>&&1
