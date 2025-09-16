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
#SBATCH --exclude=g6
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


srun python main.py --config configs/fb_computecanada_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial_online_x0_joint_onlineonly_7.3days.2.2.2.nibi.2.2.2.yaml > log.fb.computecanada.challengingandsample.pretrainrnn.balanced.lr5e6.contbest.samplercover.newd.contfreezernn.newnewd.origunet.nospatial2.ONLINE.online.online.x0.joint_onlineonly.2.2.2.2.3.2.7.2.2.2.aip.2.2.2.2.3days.2.nibi.2.2.5e6.2.2.2.debug3days 2>&1
