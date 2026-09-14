#!/bin/bash
set -euo pipefail

export PYTHONPATH="$PWD:$PWD/mmcv:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if (( $# < 1 || $# > 2 )); then
    echo "Usage: bash scripts/det_pretrain.sh OUTPUT_DIR [DATASET_CONFIG]" >&2
    exit 2
fi

DINO_PATH=${VISION_ENCODER:-microsoft/rad-dino}
OUTPUT_DIR=$1
DATASET_CONFIG=${2:-chexground/data/configs/det_pretrain.py}

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
TRAIN_BSZ=${TRAIN_BSZ:-12}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-4}
mkdir -p "$OUTPUT_DIR"

# Abnormality training is disabled; only anatomy boxes are trained.
torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-25000}" \
    -m chexground.train.train_det \
    --vis_encoder "$DINO_PATH" \
    --dataset_config "$DATASET_CONFIG" \
    --image_geometry_mode one_side_pad \
    --bf16 True \
    --tf32 True \
    --anatomy_num_queries 29 \
    --abnormality_num_classes 14 \
    --detach_anatomy_boxes_for_abnormality True \
    --teacher_feature_source backbone_frozen \
    --with_box_refine True \
    --ddetr_hidden_dim 256 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --num_feature_levels 3 \
    --decoder_n_points 8 \
    --freeze_vis_encoder True \
    --num_train_epochs 20 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --warmup_ratio 1e-4 \
    --logging_steps 5 \
    --lr_scheduler_type "cosine" \
    --do_train True \
    --do_eval True \
    --eval_strategy "steps" \
    --teacher_bce_weight 1.5 \
    --roi_bce_weight 0.01 \
    --kl_loss_weight 0.1 \
    --eval_steps 1000 \
    --eval_split_name val \
    --prediction_loss_only True \
    --per_device_train_batch_size "$TRAIN_BSZ" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --per_device_eval_batch_size "${EVAL_BSZ:-12}" \
    --train_abnormality_roi False \
    --dataloader_num_workers 16 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 5 \
    --report_to none \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"
