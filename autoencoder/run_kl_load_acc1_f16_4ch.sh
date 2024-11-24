#!/bin/bash
#SBATCH -p seas_gpu # partition (queue)
#SBATCH -n 1 # number of cores
#SBATCH --mem 60000 # memory pool for all cores
#SBATCH -t 7-0:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8

export BASE=/n/netscratch/shieber_lab/Lab/yuntian/computer/autoencoder
cd $BASE
conda activate /n/netscratch/shieber_lab/Lab/yuntian/main
which python
stdbuf -oL -eL python main_kl_f16_4ch.py --config config_kl16_4ch_lr4.5e6.yaml > log.config_kl16_4ch_lr4.5e6 2>&1
