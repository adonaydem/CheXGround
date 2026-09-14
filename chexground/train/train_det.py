# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
import pathlib
from dataclasses import dataclass, field
from typing import Optional

from chexground.openmmlab_support import ensure_openmmlab_paths
ensure_openmmlab_paths()

import transformers
from transformers import Trainer, Dinov2Config, DeformableDetrConfig

from chexground.model.ddetr import CustomDDETRConfig, CustomDDETRModel
from chexground.data.build import build_multi_datasets
from chexground.data.collator import DataCollatorForDetDataset
from chexground.data.datasets.chest_det import IMAGE_GEOMETRY_MODES


@dataclass
class ModelArguments:
    vis_encoder: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default=None)
    zs_weight_path: Optional[str] = field(default=None)
    vis_output_layer: Optional[int] = field(default=-1)  # default to the last layer
    anatomy_num_queries: Optional[int] = field(default=300)
    ddetr_hidden_dim: Optional[int] = field(default=256)
    num_encoder_layers: Optional[int] = field(default=6)
    num_decoder_layers: Optional[int] = field(default=6)
    num_feature_levels: Optional[int] = field(default=1)
    with_box_refine: Optional[bool] = field(default=True)
    abnormality_num_classes: Optional[int] = field(default=2)
    detach_anatomy_boxes_for_abnormality: Optional[bool] = field(default=True)
    teacher_feature_source: Optional[str] = field(default="backbone_frozen")
    auxiliary_loss: Optional[bool] = field(default=True)
    match_bbox_cost: Optional[int] = field(default=5)
    match_giou_cost: Optional[int] = field(default=2)
    bbox_loss_coefficient: Optional[int] = field(default=5)
    giou_loss_coefficient: Optional[int] = field(default=2)
    focal_alpha: Optional[float] = field(default=0.25)
    decoder_n_points: Optional[int] = field(default=8)
    teacher_bce_weight: Optional[float] = field(default=1.0)
    roi_bce_weight: Optional[float] = field(default=1.0)
    kl_loss_weight: Optional[float] = field(default=1.0)
    train_bbox: Optional[bool] = field(default=True)
    train_abnormality_roi: Optional[bool] = field(default=True)
    train_abnormality_teacher: Optional[bool] = field(default=True)


@dataclass
class DataArguments:
    dataset_config: str = field(default='chexground/data/configs/det_pretrain.py')
    image_geometry_mode: str = field(default="one_side_pad")
    eval_split_name: str = field(default="val")
    use_augmentation: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    freeze_vis_encoder: Optional[bool] = field(default=True)
    freeze_ddetr: Optional[bool] = field(default=False)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    lr_backbone_names: Optional[tuple] = field(default=("vis_encoder",))
    lr_linear_proj_names: Optional[tuple] = field(default=('reference_points', 'sampling_offsets'))
    lr_multiplier: Optional[float] = field(default=0.1)
    learning_rate_teacher: Optional[float] = field(default=2e-4)


def build_model(model_args):
    if model_args.model_name_or_path is not None:
        model = CustomDDETRModel.from_pretrained(model_args.model_name_or_path)
        if model_args.vis_encoder is not None:
            from transformers import Dinov2Model

            model.vis_encoder = Dinov2Model.from_pretrained(model_args.vis_encoder)
        return model

    vis_encoder_cfg = Dinov2Config.from_pretrained(model_args.vis_encoder)
    ddetr_cfg = DeformableDetrConfig(
        d_model=model_args.ddetr_hidden_dim,
        encoder_layers=model_args.num_encoder_layers,
        decoder_layers=model_args.num_decoder_layers,
        num_feature_levels=model_args.num_feature_levels,
        two_stage=False,
        two_stage_num_proposals=model_args.anatomy_num_queries,
        num_queries=model_args.anatomy_num_queries,
        num_labels=model_args.abnormality_num_classes,
        auxiliary_loss=model_args.auxiliary_loss,
        with_box_refine=model_args.with_box_refine,
        bbox_cost=model_args.match_bbox_cost,
        giou_cost=model_args.match_giou_cost,
        bbox_loss_coefficient=model_args.bbox_loss_coefficient,
        giou_loss_coefficient=model_args.giou_loss_coefficient,
        focal_alpha=model_args.focal_alpha,
        anatomy_num_queries=model_args.anatomy_num_queries,
        abnormality_num_labels=model_args.abnormality_num_classes,
        abnormality_num_classes=model_args.abnormality_num_classes,
        detach_anatomy_boxes_for_abnormality=model_args.detach_anatomy_boxes_for_abnormality,
        teacher_feature_source=model_args.teacher_feature_source,
        decoder_n_points=model_args.decoder_n_points,
        teacher_bce_weight=model_args.teacher_bce_weight,
        roi_bce_weight=model_args.roi_bce_weight,
        kl_loss_weight=model_args.kl_loss_weight,
        train_bbox=model_args.train_bbox,
        train_abnormality_roi=model_args.train_abnormality_roi,
        train_abnormality_teacher=model_args.train_abnormality_teacher,
    )
    model_cfg = CustomDDETRConfig(
        zs_weight_path=model_args.zs_weight_path,
        vis_encoder_cfg=vis_encoder_cfg,
        ddetr_cfg=ddetr_cfg,
        vis_output_layer=model_args.vis_output_layer,
    )
    return CustomDDETRModel(model_cfg, pretrained_vis_encoder=model_args.vis_encoder)


def build_optimizer(model, training_args):
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(keyword in n for keyword in training_args.lr_backbone_names) and
                       not any(keyword in n for keyword in training_args.lr_linear_proj_names) and
                       "teacher" not in n and p.requires_grad],
            "lr": training_args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(keyword in n for keyword in (*training_args.lr_backbone_names, *training_args.lr_linear_proj_names)) and
                       "teacher" not in n and p.requires_grad],
            "lr": training_args.learning_rate * training_args.lr_multiplier,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       "teacher" in n and p.requires_grad],
            "lr": training_args.learning_rate_teacher,
        }

    ]
    return torch.optim.AdamW(param_dicts, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if data_args.image_geometry_mode not in IMAGE_GEOMETRY_MODES:
        raise ValueError(
            f"image_geometry_mode must be one of {sorted(IMAGE_GEOMETRY_MODES)}, got {data_args.image_geometry_mode}."
        )
    if data_args.eval_split_name not in {"val", "test"}:
        raise ValueError(f"eval_split_name must be one of {{'val', 'test'}}, got {data_args.eval_split_name}.")
    if not training_args.do_train and not training_args.do_eval:
        raise ValueError("At least one of do_train or do_eval must be True.")

    model = build_model(model_args)
    if model_args.vis_encoder is not None:
        image_processor_name = model_args.vis_encoder
    else:
        vis_encoder_cfg = getattr(model.config, "vis_encoder_cfg", None)
        image_processor_name = getattr(vis_encoder_cfg, "_name_or_path", None)
        if not image_processor_name:
            raise ValueError(
                "vis_encoder is required for dataset preprocessing unless the loaded checkpoint config stores "
                "vis_encoder_cfg._name_or_path. Pass --vis_encoder explicitly for eval-only checkpoint runs if needed."
            )
    
    if training_args.freeze_vis_encoder:
        model.freeze_vis_encoder()
    if training_args.freeze_ddetr:
        model.freeze_ddetr()

    dataset_kwargs = dict(
        anatomy_num_queries=model_args.anatomy_num_queries,
        abnormality_num_classes=model_args.abnormality_num_classes,
        image_processor_name=image_processor_name,
        image_geometry_mode=data_args.image_geometry_mode,
        use_augmentation=data_args.use_augmentation,
    )
    train_dataset = None
    eval_dataset = None
    if training_args.do_train:
        train_dataset = build_multi_datasets(data_args.dataset_config, stage="train", **dataset_kwargs)
        if train_dataset is None:
            raise ValueError("No train datasets configured in dataset_config.")
        if training_args.do_eval:
            eval_dataset = build_multi_datasets(data_args.dataset_config, stage="val", **dataset_kwargs)
            if eval_dataset is None:
                raise ValueError("Validation evaluation is enabled, but val_datasets is not configured.")
    elif training_args.do_eval:
        eval_dataset = build_multi_datasets(data_args.dataset_config, stage=data_args.eval_split_name, **dataset_kwargs)
        if eval_dataset is None:
            raise ValueError(
                f"Eval-only run requested split={data_args.eval_split_name!r}, but no datasets are configured for it."
            )

    data_collator = DataCollatorForDetDataset()
    optimizer = build_optimizer(model, training_args) if training_args.do_train else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )

    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        trainer.save_state()
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {
                key: value.cpu()
                for key, value in state_dict.items()
            }
            del state_dict
            trainer._save(training_args.output_dir, state_dict=cpu_state_dict)

    if training_args.do_eval and not training_args.do_train:
        metric_key_prefix = "test" if data_args.eval_split_name == "test" else "eval"
        metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=metric_key_prefix)
        trainer.log_metrics(metric_key_prefix, metrics)
        trainer.save_metrics(metric_key_prefix, metrics)

if __name__ == "__main__":
    train()
