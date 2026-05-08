#!/usr/bin/env bash
set -euo pipefail

# Caso precise refazer o Docker
# docker build -t tf-train .
touch classes.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run -it --rm --runtime=nvidia --gpus all \
  -p 6006:6006 \
  -v "$SCRIPT_DIR/models:/app/models" \
  -v "$SCRIPT_DIR/data:/app/data" \
  -v "$SCRIPT_DIR/logs:/app/logs" \
  -v "$SCRIPT_DIR/classes.txt:/app/classes.txt" \
  -v "$SCRIPT_DIR/build.py:/app/build.py:ro" \
  tf-train