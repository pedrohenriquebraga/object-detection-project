#!/usr/bin/env bash
set -euo pipefail

# Caso precise refazer o Docker
# sudo docker build -t tf-train .
touch classes.txt

sudo docker --context default run -it --rm --runtime=nvidia --gpus all \
  -p 6006:6006 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/classes.txt:/app/classes.txt" \
  -v "$(pwd)/build.py:/app/build.py:ro" \
  tf-train

# TensorBoard (em outro terminal):
# sudo docker --context default run -it --rm \
#   -p 6006:6006 \
#   -v "$(pwd)/logs:/app/logs" \
#   tf-train tensorboard --logdir /app/logs/fit --bind_all --port 6006