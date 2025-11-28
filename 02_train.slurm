#!/bin/bash
#SBATCH --job-name=kaggle-csiro-train
#SBATCH --gpus=2
#SBATCH --output=slurm_logs/kaggle-csiro-train_%j.out

EXP_NAME=$1

# Check if EXP_NAME is provided
if [ -z "$EXP_NAME" ]; then
    echo "Error: EXP_NAME is required. Usage: sbatch 02_train.slurm <exp_name>"
    exit 1
fi

srun docker run --gpus all --rm \
-e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
--name ${EXP_NAME}_train \
--shm-size=8gb \
-v $(pwd):/app \
-w /app \
-e WANDB_API_KEY=${WANDB_API_KEY} \
-e KAGGLE_USERNAME=${KAGGLE_USERNAME} \
-e KAGGLE_KEY=${KAGGLE_KEY} \
shuto.goya/kaggle-csiro:latest uv run src/${EXP_NAME}/train.py