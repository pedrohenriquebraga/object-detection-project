#!/usr/bin/env bash
set -euo pipefail

# Script para converter modelo .keras para .tflite
# Para EfficientNet, 'dynamic' e 'float' são mais estáveis.
# Uso: ./convert.sh [quantization_mode] [runtime]
#   quantization_mode: int8, float16, dynamic (padrão), float
#   runtime: builtin (padrão, sem Flex), flex, auto

QUANTIZATION=${1:-float16}
RUNTIME=${2:-auto}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run -it --rm \
  -v "$SCRIPT_DIR/models:/app/models" \
  -v "$SCRIPT_DIR/data:/app/data" \
  -v "$SCRIPT_DIR/convert.py:/app/convert.py:ro" \
  tf-train python3 /app/convert.py --quantization "$QUANTIZATION" --runtime "$RUNTIME"
