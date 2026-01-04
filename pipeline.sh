#!/bin/bash

EXP_NAME=$1

uv run ./src/$EXP_NAME/create_dataset.py --config-path ./config/$EXP_NAME.yaml
uv run ./src/$EXP_NAME/train.py --folds 0 1 2 3 4
uv run ./src/$EXP_NAME/evaluation.py \
    --device mps \
    --model_dir ./output/$EXP_NAME
uv run ./src/$EXP_NAME/inference.py \
    --device auto \
    --batch-size 256 \
    --num-workers 4 \
    --config-path ./config/$EXP_NAME.yaml \
    --model-dir ./output/$EXP_NAME \
    --output-dir ./output/$EXP_NAME
uv run ./src/utils/upload_scripts.py --exp-name $EXP_NAME