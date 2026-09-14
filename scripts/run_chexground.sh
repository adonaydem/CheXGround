#!/bin/bash
set -euo pipefail

export PYTHONPATH="$PWD:$PWD/mmcv:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ "$1" == --model || "$1" == --merged-model ]]; then
    MODEL_ARGS=(--base-model "$2")
else
    MODEL_ARGS=(--base-model "$1" --lora-path "$2")
fi
INPUT_PATH=$3
shift 3

OUTPUT_ARGS=()

if [[ $# -gt 0 && "$1" != --* ]]; then
    OUTPUT_ARGS=(--output "$1")
    shift
fi

python -m chexground.eval.run_chexground \
    "${MODEL_ARGS[@]}" \
    --input "$INPUT_PATH" \
    "${OUTPUT_ARGS[@]}" "$@"
