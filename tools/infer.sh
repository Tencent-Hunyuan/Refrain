#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${NPROC_PER_NODE:-8}" != 8 ]]; then
    echo 'Inference requires one local group of 8 GPUs (SP=8).' >&2
    exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
    echo 'Create the repository .venv using docs/INSTALL.md first.' >&2
    exit 2
fi
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 tools/infer.py "$@"
