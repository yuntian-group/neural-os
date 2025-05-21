CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python preprocess_dataset.py --start_idx 0 --end_idx 3000 > log.preprocess_dataset.0_3k 2>&1 &
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python preprocess_dataset.py --start_idx 3000 --end_idx 6000 > log.preprocess_dataset.3k_6k 2>&1 &
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python preprocess_dataset.py --start_idx 6000 --end_idx 9000 > log.preprocess_dataset.6k_9k 2>&1 &
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python preprocess_dataset.py --start_idx 9000 --end_idx 12000 > log.preprocess_dataset.9k_12k 2>&1 &
CUDA_VISIBLE_DEVICES=4 stdbuf -oL -eL python preprocess_dataset.py --start_idx 12000 --end_idx 15000 > log.preprocess_dataset.12k_15k 2>&1 &
CUDA_VISIBLE_DEVICES=5 stdbuf -oL -eL python preprocess_dataset.py --start_idx 15000 --end_idx 18000 > log.preprocess_dataset.15k_18k 2>&1 &
CUDA_VISIBLE_DEVICES=6 stdbuf -oL -eL python preprocess_dataset.py --start_idx 18000 --end_idx 21000 > log.preprocess_dataset.18k_21k 2>&1 &
CUDA_VISIBLE_DEVICES=7 stdbuf -oL -eL python preprocess_dataset.py --start_idx 21000 --end_idx 24000 > log.preprocess_dataset.21k_24k 2>&1 &



CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python preprocess_dataset.py --start_idx 0 --end_idx 5000 > log.preprocess_dataset.0_5k 2>&1 &
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python preprocess_dataset.py --start_idx 5000 --end_idx 10000 > log.preprocess_dataset.5k_10k 2>&1 &
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python preprocess_dataset.py --start_idx 10000 --end_idx 15000 > log.preprocess_dataset.10k_15k 2>&1 &
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python preprocess_dataset.py --start_idx 15000 --end_idx 20000 > log.preprocess_dataset.15k_20k 2>&1 &
CUDA_VISIBLE_DEVICES=4 stdbuf -oL -eL python preprocess_dataset.py --start_idx 20000 --end_idx 25000 > log.preprocess_dataset.20k_25k 2>&1 &
CUDA_VISIBLE_DEVICES=5 stdbuf -oL -eL python preprocess_dataset.py --start_idx 25000 --end_idx 30000 > log.preprocess_dataset.25k_30k 2>&1 &
CUDA_VISIBLE_DEVICES=6 stdbuf -oL -eL python preprocess_dataset.py --start_idx 30000 --end_idx 35000 > log.preprocess_dataset.30k_35k 2>&1 &
CUDA_VISIBLE_DEVICES=7 stdbuf -oL -eL python preprocess_dataset.py --start_idx 35000 --end_idx 40000 > log.preprocess_dataset.35k_40k 2>&1 &



python preprocess_dataset.py --start_idx 0 --end_idx 1000 --batch_size 8
