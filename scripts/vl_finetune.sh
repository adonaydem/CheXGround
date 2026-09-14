#!/bin/bash
set -euo pipefail

export PYTHONPATH="$PWD:$PWD/mmcv:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MODEL_PATH=$1
OUTPUT_DIR=$2
DATASET_CONFIG=${3:-chexground/data/configs/chexground_finetune.py}

NPROC_PER_NODE=${NPROC_PER_NODE:-2}
TRAIN_BSZ=${TRAIN_BSZ:-2}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-8}
mkdir -p "$OUTPUT_DIR"

torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-25002}" \
    -m chexground.train.train_mem \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_config "$DATASET_CONFIG" \
    --freeze_visual True \
    --freeze_llm True \
    --freeze_llm_keep_added_tokens_trainable True \
    --train_vl_projectors True \
    --train_tac_projector True \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --bf16 True \
    --tf32 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 2 \
    --per_device_train_batch_size "$TRAIN_BSZ" \
    --per_device_eval_batch_size "${EVAL_BSZ:-2}" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --eval_strategy steps \
    --eval_steps 5000 \
    --save_strategy steps \
    --save_steps 3000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-False}" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --report_to none \
    --dataloader_num_workers 8 \
    --group_by_data_source False \
    --max_grad_norm 1.0 \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"
