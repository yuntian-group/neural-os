#!/bin/bash
#SBATCH -p seas_gpu
#SBATCH -n 1
#SBATCH --mem 60000
#SBATCH -t 7-0:00
#SBATCH -o slurm.%N.%j.1.out
#SBATCH -e slurm.%N.%j.1.err
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8

export BASE=/n/netscratch/shieber_lab/Lab/yuntian/computer/autoencoder
cd $BASE
conda activate /n/netscratch/shieber_lab/Lab/yuntian/main
which python
stdbuf -oL -eL python preprocess_dataset.py --start_idx 18000 --end_idx 24000 > log.preprocess_dataset.512_384.4 2>&1 
