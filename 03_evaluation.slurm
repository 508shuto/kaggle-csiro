#!/bin/bash
#SBATCH --job-name=kaggle-csiro-evaluation
#SBATCH --gpus=2
#SBATCH --output=slurm_logs/kaggle-csiro-evaluation_%j.out

EXP_NAME=$1
FOLD=$2
# if fold is null, then run all folds
if [ -z "$FOLD" ]; then
    FOLD="0 1 2 3 4"
fi

srun docker run --gpus all --rm \
-e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
--name ${EXP_NAME}_evaluation \
--shm-size=8gb \
-v $(pwd):/app \
-w /app \
-e WANDB_API_KEY=${WANDB_API_KEY} \
-e KAGGLE_USERNAME=${KAGGLE_USERNAME} \
-e KAGGLE_KEY=${KAGGLE_KEY} \
shuto.goya/kaggle-csiro:latest uv run src/${EXP_NAME}/evaluation.py --model_dir ./output/${EXP_NAME}
