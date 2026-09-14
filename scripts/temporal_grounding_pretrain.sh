#!/bin/bash
set -euo pipefail

export PYTHONPATH="$PWD:$PWD/mmcv:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DDETR_CHECKPOINT=$1
OUTPUT_DIR=$2
REGION_JSON_ROOT=$3
DATASET_CONFIG=$4
TEXT_ENCODER_NAME=$5
VIS_ENCODER_NAME=${6:-${VISION_ENCODER:-microsoft/rad-dino}}

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
TRAIN_BSZ=${TRAIN_BSZ:-64}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
mkdir -p "$OUTPUT_DIR"

# Abnormality training is disabled with cls_weight set to 0.
torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-25001}" \
    -m chexground.train.train_temporal_grounding \
    --ddetr_checkpoint "$DDETR_CHECKPOINT" \
    --text_encoder_name "$TEXT_ENCODER_NAME" \
    --vis_encoder_name "$VIS_ENCODER_NAME" \
    --temporal_use_libra_prior_bias True \
    --region_json_root "$REGION_JSON_ROOT" \
    --dataset_config "$DATASET_CONFIG" \
    --image_geometry_mode one_side_pad \
    --local_aux_type attn_kl \
    --attn_kl_soft_alpha 0.5 \
    --temp1 4.0 \
    --temp2 5.0 \
    --temp3 10.0 \
    --roi_output_size 7 \
    --roi_sampling_ratio 2 \
    --cls_weight 0.0 \
    --local_gloria_weight 1.0 \
    --local_aux_weight 0.3 \
    --global_contrastive_weight 1.0 \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 45 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --warmup_ratio 1e-3 \
    --logging_steps 5 \
    --lr_scheduler_type "cosine" \
    --do_train True \
    --do_eval True \
    --eval_strategy "steps" \
    --eval_steps 2000 \
    --eval_split_name val \
    --load_best_model_at_end True \
    --metric_for_best_model eval_recall_mean \
    --greater_is_better True \
    --early_stopping_patience 50 \
    --per_device_train_batch_size "$TRAIN_BSZ" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --per_device_eval_batch_size "${EVAL_BSZ:-16}" \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 5 \
    --report_to none \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"
