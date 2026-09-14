#!/bin/bash
set -euo pipefail

export PYTHONPATH="$PWD:$PWD/mmcv:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if (( $# < 3 || $# > 4 )); then
    echo "Usage: bash scripts/vl_pretrain.sh PRETRAINED_VLM S1_CHECKPOINT OUTPUT_DIR [DATASET_CONFIG]" >&2
    exit 2
fi

LIBRA_PATH=$1
GRPA_PATH=$2
VISION_TOWER_OVERRIDE=${VISION_ENCODER:-microsoft/rad-dino}
OUTPUT_DIR=$3
DATASET_CONFIG=${4:-chexground/data/configs/chexground_pretrain.py}

NPROC_PER_NODE=${NPROC_PER_NODE:-2}
TRAIN_BSZ=${TRAIN_BSZ:-3}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-4}
mkdir -p "$OUTPUT_DIR"

torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-25005}" \
    -m chexground.train.train_mem \
    --libra "$LIBRA_PATH" \
    --grpa "$GRPA_PATH" \
    --vision_tower_override "$VISION_TOWER_OVERRIDE" \
    --reset_tac_projector True \
    --tokenizer_max_temporal_frames 4 \
    --dataset_config "$DATASET_CONFIG" \
    --freeze_visual True \
    --group_by_data_source False \
    --freeze_llm True \
    --freeze_llm_keep_added_tokens_trainable True \
    --bf16 True \
    --tf32 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "$TRAIN_BSZ" \
    --per_device_eval_batch_size "${EVAL_BSZ:-3}" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --eval_strategy steps \
    --eval_steps 3000 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 3e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-False}" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --report_to none \
    --dataloader_num_workers 0 \
    --dataloader_pin_memory False \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"
