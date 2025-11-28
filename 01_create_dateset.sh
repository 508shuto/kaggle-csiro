#!/bin/bash

EXP_NAME=$1

docker run -it --rm \
-v $(pwd):/app \
-w /app \
shuto.goya/kaggle-csiro:latest uv run src/${EXP_NAME}/create_dataset.py