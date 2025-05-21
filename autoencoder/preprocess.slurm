#!/bin/bash
#SBATCH --account=def-yuntian
#SBATCH --job-name=preproc_ds
#SBATCH --output=logs/preproc_%A_%a.out
#SBATCH --error=logs/preproc_%A_%a.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --array=0-9

# calculate start/end indices
START=$(( SLURM_ARRAY_TASK_ID * 1000 ))
END=$(( START + 1000 ))

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID: start=$START, end=$END"
cd /home/yuntian/projects/def-yuntian/yuntian/computer/autoencoder
python preprocess_dataset.py \
    --start_idx "${START}" \
    --end_idx   "${END}" \
    --batch_size 8
