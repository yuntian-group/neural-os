#!/bin/bash
#SBATCH --account=rrg-yuntian
#SBATCH --job-name=train_ds_1gpu
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@90
#SBATCH --open-mode=append
#SBATCH --exclude=g4
#
source ~/.bashrc

cd /home/yuntian/projects/rrg-yuntian/yuntian/computer/computer

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

srun python main.py --config configs/fix_8gpu_bug_1gpu.2.yaml > log.fix_8gpu_bug_1gpu.2.2 2>&1
