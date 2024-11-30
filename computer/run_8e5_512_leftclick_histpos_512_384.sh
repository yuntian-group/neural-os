#!/bin/bash
#SBATCH -p seas_gpu # partition (queue)
#SBATCH -n 1 # number of cores
#SBATCH --mem 180000 # memory pool for all cores
#SBATCH -t 7-0:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8

export BASE=/n/netscratch/shieber_lab/Lab/yuntian/computer/computer
cd $BASE
conda activate /n/netscratch/shieber_lab/Lab/yuntian/main
which python
stdbuf -oL -eL python main.py --config configs/pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384.yaml > log.pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384.cont2 2>&1
