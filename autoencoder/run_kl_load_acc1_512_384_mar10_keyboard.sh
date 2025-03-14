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
#stdbuf -oL -eL python main_kl_512_384_mar10_keyboard.py --config config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard.yaml > log.ae.kl.bsz8_acc3_lr4.5e6_load_acc1.save.512_384.mar10.keyboard 2>&1


stdbuf -oL -eL python main_kl_512_384_mar10_keyboard.py --config config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard_contlr1e6.yaml --from_ckpt saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384_mar10_keyboard/model-585000.ckpt > log.ae.kl.bsz8_acc3_lr4.5e6_load_acc1.save.512_384.mar10.keyboard.cont 2>&1




# init, 4 channels
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python main_kl_512_384_mar10_keyboard_init.py --config config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard_init_4.yaml > log.init.ae.kl.bsz8_acc3_lr4.5e6_load_acc1.save.512_384.mar10.keyboard.4 2>&1&
# init, 8 channels
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python main_kl_512_384_mar10_keyboard_init.py --config config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard_init_8.yaml > log.init.ae.kl.bsz8_acc3_lr4.5e6_load_acc1.save.512_384.mar10.keyboard.8 2>&1&
# init, 16 channels
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python main_kl_512_384_mar10_keyboard_init.py --config config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard_init_16.yaml > log.init.ae.kl.bsz8_acc3_lr4.5e6_load_acc1.save.512_384.mar10.keyboard.16 2>&1&


# cont
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python main_kl_512_384_mar10_keyboard.py --config config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard_contlr1e6.yaml --from_ckpt saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384_mar10_keyboard_cont/model-741000.ckpt > log.ae.kl.bsz8_acc3_lr4.5e6_load_acc1.save.512_384.mar10.keyboard.cont2 2>&1
