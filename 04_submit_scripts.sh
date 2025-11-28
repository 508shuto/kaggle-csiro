#!/bin/bash

EXP_NAME=$1

docker run -it --rm \
-v $(pwd):/app \
-w /app \
-e WANDB_API_KEY=${WANDB_API_KEY} \
-e KAGGLE_USERNAME=${KAGGLE_USERNAME} \
-e KAGGLE_KEY=${KAGGLE_KEY} \
shuto.goya/kaggle-csiro:latest uv run src/utils/upload_scripts.py --exp_name ${EXP_NAME}