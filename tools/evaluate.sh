#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
EVAL_PYTHON="${EVAL_PYTHON:-.venv-eval/bin/python}"
if [[ ! -x "$EVAL_PYTHON" ]]; then
    echo "Missing evaluation environment: follow docs/EVALUATION.md" >&2
    exit 1
fi
for stage in transition clip semantic regions ta reference_semantic sdc; do
    "$EVAL_PYTHON" -m evaluation.run worker --stage "$stage" "$@"
done
"$EVAL_PYTHON" -m evaluation.run summarize "$@"
