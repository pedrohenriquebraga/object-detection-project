# TensorBoard (em outro terminal):
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker run -it --rm \
   -p 8081:8081 \
   -v "$SCRIPT_DIR/logs:/app/logs" \
   tf-train tensorboard --logdir /app/logs/fit --bind_all --port 8081