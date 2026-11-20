#!/usr/bin/env bash
# Usage: tools/train.sh <text_to_video|reference_training|variable_duration_training> [--resume DIR] [--output DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
exec torchrun --nnodes="${NNODES:-1}" --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" tools/train.py "$@"
