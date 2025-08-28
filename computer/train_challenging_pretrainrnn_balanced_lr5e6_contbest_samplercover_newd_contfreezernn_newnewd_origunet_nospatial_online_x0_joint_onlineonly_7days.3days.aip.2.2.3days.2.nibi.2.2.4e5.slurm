#!/bin/bash
#SBATCH --account=rrg-yuntian
#SBATCH --job-name=train_ds
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --gres=gpu:h100:8
#SBATCH --time=0-12:00:00

source ~/.bashrc

cd /home/yuntian/projects/rrg-yuntian/yuntian/computer/computer
# Important: Set CUDA memory allocator explicitly
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# If your PyTorch is >= 2.0, you can add (but it's optional, remove if unsupported)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

nvidia-smi

srun python main.py --config configs/fb_computecanada_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial_online_x0_joint_onlineonly_7.3days.2.2.2.nibi.2.2.4e5.yaml > log.fb.computecanada.challengingandsample.pretrainrnn.balanced.lr5e6.contbest.samplercover.newd.contfreezernn.newnewd.origunet.nospatial2.ONLINE.online.online.x0.joint_onlineonly.2.2.2.2.3.2.7.2.2.2.aip.2.2.2.2.3days.2.nibi.2.2.4e5 2>&1
