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

python main.py --config configs/fb_diffusion_freezernn_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_lr2e5_context64_computecanada_challengingandsample_pretrainrnn_balanced_lr8e5.yaml > log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont.lr2e5.context64.computecanada.challengingandsample.pretrainrnn.balanced.lr8e5 2>&1
