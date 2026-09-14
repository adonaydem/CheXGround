#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd):$(pwd)/mmcv:${PYTHONPATH:-}"



LIBRA_PATH=${1:-/path/to/libra-checkpoint}
GRPA_PATH=${2:-/path/to/grpa-checkpoint}
VISION_TOWER_OVERRIDE=${3:-/path/to/vision-encoder}
OUTPUT_DIR=${4:-/path/to/output}
DATASET_CONFIG=${5:-chexground/data/configs/chexground_pretrain.py}
EVAL_STEPS=${6:-1000}
RESET_TAC_PROJECTOR=${7:-True}
TOKENIZER_MAX_TEMPORAL_FRAMES=${8:-4}

# ===== SYSTEM SETTINGS =====
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-25004}


# ===== VALIDATION =====
if [[ ! -d "$LIBRA_PATH" ]]; then
    echo "Model checkpoint directory not found: $LIBRA_PATH" >&2
    exit 1
fi

if [[ ! -d "$GRPA_PATH" ]]; then
    echo "GRPA checkpoint directory not found: $GRPA_PATH" >&2
    exit 1
fi
if [[ ! -f "$DATASET_CONFIG" ]]; then
    echo "Dataset config not found: $DATASET_CONFIG" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ===== RUN =====
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    -m chexground.train.train_mem \
    --libra "$LIBRA_PATH" \
    --grpa "$GRPA_PATH" \
    --vision_tower_override "$VISION_TOWER_OVERRIDE" \
    --reset_tac_projector "$RESET_TAC_PROJECTOR" \
    --tokenizer_max_temporal_frames "$TOKENIZER_MAX_TEMPORAL_FRAMES" \
    --dataset_config "$DATASET_CONFIG" \
    --freeze_visual True \
    --freeze_llm True \
    --bf16 True \
    --tf32 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --deepspeed Libra/scripts/zero3.json \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 12 \
    --eval_strategy steps \
    --eval_steps "$EVAL_STEPS" \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --model_max_length 2048 \
    --report_to none \
    --dataloader_num_workers 8 \
    2>&1 | tee "$OUTPUT_DIR/train.log"
