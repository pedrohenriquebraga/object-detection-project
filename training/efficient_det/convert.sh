#!/usr/bin/env bash
set -euo pipefail

# Script para converter modelo .keras para .tflite
# Para EfficientNet, 'dynamic' e 'float' são mais estáveis.
# Uso: ./convert.sh [quantization_mode] [runtime]
#   quantization_mode: int8, float16, dynamic (padrão), float
#   runtime: builtin (padrão, sem Flex), flex, auto

QUANTIZATION=${1:-dynamic}
RUNTIME=${2:-builtin}

sudo docker --context default run -it --rm \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/convert.py:/app/convert.py:ro" \
  tf-train python3 /app/convert.py --quantization "$QUANTIZATION" --runtime "$RUNTIME"
