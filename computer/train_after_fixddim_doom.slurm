#!/bin/bash
#SBATCH --account=rrg-yuntian
#SBATCH --job-name=train_ds
#SBATCH --output=logs_sep/train_%j.out
#SBATCH --error=logs_sep/train_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --mem=2000G
#SBATCH --exclusive
#SBATCH --gres=gpu:h100:8
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@90
#SBATCH --open-mode=append
#
source ~/.bashrc

cd /home/yuntian/projects/rrg-yuntian/yuntian/computer/computer

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# Reasonable defaults for multi-GPU stability
export OMP_NUM_THREADS=6
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_BLOCKING_WAIT=0
df -h /dev/shm
ls -lh /dev/shm | head
# If it's safe and only your job owns the torch shm files:
rm -f /dev/shm/torch_*


srun python main.py --config configs/train_after_fixddim_doom.yaml > log.train_after_fixddim_doom 2>&1
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/train_after_fixddim_doom.yaml > log.train_after_fixddim_doom 2>&1
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/train_after_fixddim_doom.yaml > log.train_after_fixddim_doom.rerun.rerun.rerun 2>&1

CUDA_VISIBLE_DEVICES=1 python main.py --config configs/train_after_fixddim_doom_play.yaml > log.train_after_fixddim_doom.rerun.rerun.rerun.play 2>&1
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python main.py --config configs/train_after_fixddim_doom_play.yaml > log.train_after_fixddim_doom.rerun.rerun.rerun.play 2>&1
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python main.py --config configs/train_after_fixddim_doom_play.7gpus.yaml > log.train_after_fixddim_doom.rerun.rerun.rerun.play.7gpus 2>&1
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python main.py --config configs/train_after_fixddim_doom_play.7gpus.yaml > log.train_after_fixddim_doom.rerun.rerun.rerun.play.7gpus.classicdoom 2>&1
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python main.py --config configs/train_after_fixddim_doom_play.7gpus.yaml > log.train_after_fixddim_doom.rerun.rerun.rerun.play.7gpus.classicdoom.fixarrow 2>&1
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python main.py --config configs/train_after_fixddim_doom_play.7gpus.yaml > log.train_after_fixddim_doom.rerun.rerun.rerun.play.7gpus.classicdoom.fixarrow.2.fix2 2>&1
