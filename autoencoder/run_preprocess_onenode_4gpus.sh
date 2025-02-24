CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python preprocess_dataset.py --start_idx 0 --end_idx 6000 > log.preprocess_dataset.0_6k 2>&1&
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python preprocess_dataset.py --start_idx 6000 --end_idx 12000 > log.preprocess_dataset.6k_12k 2>&1&
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python preprocess_dataset.py --start_idx 12000 --end_idx 18000 > log.preprocess_dataset.12k_18k 2>&1&
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python preprocess_dataset.py --start_idx 18000 --end_idx 24000 > log.preprocess_dataset.18k_24k 2>&1&

