#!/usr/bin/env bash
set -euo pipefail

# sudo docker build -t tf-train .
sudo docker --context default run -it --rm --runtime=nvidia --gpus all \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/build.py:/app/build.py:ro" \
  tf-train