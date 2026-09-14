import pathlib
import inspect
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from chexground.openmmlab_support import ensure_openmmlab_paths

ensure_openmmlab_paths()

import transformers
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, Trainer
from transformers.trainer import has_length, is_datasets_available, seed_worker

from chexground.data.build import build_multi_datasets
from chexground.data.collator_temporal_grounding import DataCollatorForTemporalGroundingDataset
from chexground.data.temporal_batching import TemporalLengthBatchSampler, has_temporal_lengths
from chexground.model.ddetr import CustomDDETRConfig, CustomDDETRModel
from chexground.model.temporal_grounding import GRPAConfig, GRPA

if is_datasets_available():
    import datasets
else:
    datasets = None


def _compute_local_image_recall_sums(
    region_embeddings: torch.Tensor,
    phrase_embeddings: torch.Tensor,
    roi_valid_mask: torch.BoolTensor,
    phrase_batch_index: torch.LongTensor,
    phrase_positive_roi_mask: torch.BoolTensor,
) -> torch.Tensor:
    batch_size = region_embeddings.shape[0]
    if phrase_embeddings.shape[0] == 0:
        return torch.stack([region_embeddings.new_tensor(0.0), region_embeddings.new_tensor(0.0)])

    roi_valid_mask = roi_valid_mask.to(dtype=torch.bool, device=region_embeddings.device)
    phrase_batch_index = phrase_batch_index.to(device=region_embeddings.device, dtype=torch.long)
    phrase_positive_roi_mask = phrase_positive_roi_mask.to(device=region_embeddings.device, dtype=torch.bool)

    image_recall_sum = region_embeddings.new_tensor(0.0)
    image_count = region_embeddings.new_tensor(0.0)

    for image_index in range(batch_size):
        phrase_mask = phrase_batch_index == image_index
        if not phrase_mask.any():
            continue
        valid_roi_mask = roi_valid_mask[image_index]
        if not valid_roi_mask.any():
            continue

        image_phrase_embeddings = phrase_embeddings[phrase_mask]
        image_region_embeddings = region_embeddings[image_index, valid_roi_mask]
        positive_mask = phrase_positive_roi_mask[phrase_mask][:, valid_roi_mask].transpose(0, 1)
        supervised_roi_mask = positive_mask.any(dim=1)
        if not supervised_roi_mask.any():
            continue

        roi_to_phrase_scores = torch.matmul(image_region_embeddings, image_phrase_embeddings.transpose(0, 1))
        topk = roi_to_phrase_scores.topk(k=min(1, roi_to_phrase_scores.shape[1]), dim=1).indices
        hits = positive_mask.gather(1, topk).any(dim=1).float()
        image_recall_sum = image_recall_sum + hits[supervised_roi_mask].mean()
        image_count = image_count + 1.0

    return torch.stack([image_recall_sum, image_count])


class TemporalGroundingTrainer(Trainer):
    def _filter_model_inputs(self, model, inputs):
        actual_model = model.module if hasattr(model, "module") else model
        allowed_keys = getattr(self, "_cached_model_input_keys", None)
        if allowed_keys is None:
            allowed_keys = set(inspect.signature(actual_model.forward).parameters.keys())
            self._cached_model_input_keys = allowed_keys
        return {key: value for key, value in inputs.items() if key in allowed_keys}

    def _get_temporal_batch_dataloader(
        self,
        dataset,
        description: str,
        batch_size: int,
        shuffle: bool,
        is_training: bool = False,
    ) -> DataLoader:
        data_collator = self.data_collator
        if datasets is not None and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

        batch_sampler = TemporalLengthBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=self.args.dataloader_drop_last if is_training else False,
        )
        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if is_training:
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    def get_train_dataloader(self) -> DataLoader:
        if not (has_length(self.train_dataset) and has_temporal_lengths(self.train_dataset)):
            return super().get_train_dataloader()
        return self._get_temporal_batch_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size,
            shuffle=True,
            is_training=True,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        if not (eval_dataset is not None and has_length(eval_dataset) and has_temporal_lengths(eval_dataset)):
            return super().get_eval_dataloader(eval_dataset=eval_dataset)
        return self._get_temporal_batch_dataloader(
            dataset=eval_dataset,
            description="Evaluation",
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            is_training=False,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        filtered_inputs = self._filter_model_inputs(model, inputs)
        outputs = model(**filtered_inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        if not torch.isfinite(loss):
            raise FloatingPointError
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()

        total_loss = 0.0
        total_examples = 0.0
        total_global_r1 = 0.0
        total_global_count = 0.0
        total_local_r1 = 0.0
        total_local_count = 0.0

        for inputs in eval_dataloader:
            inputs = self._prepare_inputs(inputs)
            batch_size = float(inputs["images"].shape[0])
            with torch.no_grad():
                outputs = model(**self._filter_model_inputs(model, inputs))

            loss_tensor = outputs.loss
            if not torch.isfinite(loss_tensor):
                raise FloatingPointError
            batch_loss = loss_tensor.detach() * batch_size
            total_loss += float(self.accelerator.gather_for_metrics(batch_loss.reshape(1)).sum().item())
            total_examples += float(
                self.accelerator.gather_for_metrics(batch_loss.new_tensor(batch_size).reshape(1)).sum().item()
            )

            global_valid_mask = inputs["report_valid_mask"]
            global_pair_logits = outputs.global_pair_logits.detach()
            valid_mask = global_valid_mask.to(dtype=torch.bool, device=global_pair_logits.device)
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
            global_r1 = global_pair_logits.new_tensor(0.0)
            if valid_indices.numel() > 0:
                logits = global_pair_logits.index_select(0, valid_indices).index_select(1, valid_indices)
                # Keep topk tie behavior for the disabled global path's compatibility scores.
                topk = logits.topk(k=min(1, logits.shape[1]), dim=1).indices
                targets = torch.arange(logits.shape[0], device=logits.device, dtype=topk.dtype).unsqueeze(1)
                global_r1 = topk.eq(targets).any(dim=1).float().sum()
            global_count = global_valid_mask.float().sum()

            local_r1_sum, local_count = _compute_local_image_recall_sums(
                region_embeddings=outputs.region_embeddings.detach(),
                phrase_embeddings=outputs.text_embeddings.detach(),
                roi_valid_mask=outputs.roi_valid_mask,
                phrase_batch_index=inputs["phrase_batch_index"],
                phrase_positive_roi_mask=inputs["phrase_positive_roi_mask"],
            )

            total_global_r1 += float(self.accelerator.gather_for_metrics(global_r1.reshape(1)).sum().item())
            total_global_count += float(
                self.accelerator.gather_for_metrics(global_count.to(dtype=batch_loss.dtype).reshape(1)).sum().item()
            )
            total_local_r1 += float(self.accelerator.gather_for_metrics(local_r1_sum.reshape(1)).sum().item())
            total_local_count += float(
                self.accelerator.gather_for_metrics(local_count.to(dtype=batch_loss.dtype).reshape(1)).sum().item()
            )

        eval_loss = total_loss / max(total_examples, 1.0)
        global_r1 = total_global_r1 / max(total_global_count, 1.0)
        local_r1 = total_local_r1 / max(total_local_count, 1.0)
        metrics = {
            f"{metric_key_prefix}_loss": eval_loss,
            f"{metric_key_prefix}_global_r@1": global_r1,
            f"{metric_key_prefix}_local_r@1": local_r1,
            f"{metric_key_prefix}_recall_mean": 0.5 * (global_r1 + local_r1),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


@dataclass
class ModelArguments:
    ddetr_checkpoint: str = field(default=None)
    text_encoder_name: str = field(default="microsoft/BiomedVLP-BioViL-T")
    vis_encoder_name: Optional[str] = field(default=None)
    num_classes: int = field(default=14)
    roi_output_size: int = field(default=7)
    roi_sampling_ratio: int = field(default=2)
    roi_num_heads: int = field(default=8)
    global_num_heads: int = field(default=8)
    roi_mlp_ratio: float = field(default=4.0)
    global_mlp_ratio: float = field(default=4.0)
    jitter_translate: float = field(default=0.05)
    jitter_scale: float = field(default=0.05)
    jitter_expand_max: float = field(default=0.3)
    text_max_length: int = field(default=96)
    cls_weight: float = field(default=0.0)
    local_gloria_weight: float = field(default=1.0)
    local_aux_weight: float = field(default=1.0)
    global_contrastive_weight: float = field(default=1.0)
    local_aux_type: str = field(default="cosine")
    attn_kl_soft_alpha: float = field(default=0.8)
    temp1: float = field(default=4.0)
    temp2: float = field(default=5.0)
    temp3: float = field(default=10.0)
    temporal_num_heads: Optional[int] = field(default=None)
    temporal_mlp_ratio: Optional[float] = field(default=None)
    temporal_use_libra_prior_bias: bool = field(default=True)


@dataclass
class DataArguments:
    dataset_config: str = field(default="chexground/data/configs/temporal_grounding_mimic.py")
    region_json_root: str = field(default=None)
    image_geometry_mode: str = field(default="one_side_pad")
    eval_split_name: str = field(default="val")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    save_safetensors: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    early_stopping_patience: int = field(default=8)
    early_stopping_threshold: float = field(default=0.0)


def build_model(model_args: ModelArguments):
    ddetr_cfg = CustomDDETRConfig.from_pretrained(model_args.ddetr_checkpoint)
    text_encoder_cfg = transformers.AutoConfig.from_pretrained(
        model_args.text_encoder_name,
        trust_remote_code=True,
    )
    config = GRPAConfig(
        ddetr_checkpoint=model_args.ddetr_checkpoint,
        ddetr_cfg=ddetr_cfg,
        text_encoder_name=model_args.text_encoder_name,
        text_encoder_cfg=text_encoder_cfg,
        num_classes=model_args.num_classes,
        roi_output_size=model_args.roi_output_size,
        roi_sampling_ratio=model_args.roi_sampling_ratio,
        roi_num_heads=model_args.roi_num_heads,
        global_num_heads=model_args.global_num_heads,
        roi_mlp_ratio=model_args.roi_mlp_ratio,
        global_mlp_ratio=model_args.global_mlp_ratio,
        jitter_translate=model_args.jitter_translate,
        jitter_scale=model_args.jitter_scale,
        jitter_expand_max=model_args.jitter_expand_max,
        text_max_length=model_args.text_max_length,
        cls_weight=model_args.cls_weight,
        local_gloria_weight=model_args.local_gloria_weight,
        local_aux_weight=model_args.local_aux_weight,
        global_contrastive_weight=model_args.global_contrastive_weight,
        local_aux_type=model_args.local_aux_type,
        attn_kl_soft_alpha=model_args.attn_kl_soft_alpha,
        temp1=model_args.temp1,
        temp2=model_args.temp2,
        temp3=model_args.temp3,
        temporal_num_heads=model_args.temporal_num_heads,
        temporal_mlp_ratio=model_args.temporal_mlp_ratio,
        temporal_use_libra_prior_bias=model_args.temporal_use_libra_prior_bias,
    )
    model = GRPA(config)
    # Restore frozen components in place without advancing training's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        model.detector.load_state_dict(CustomDDETRModel.from_pretrained(model_args.ddetr_checkpoint).state_dict())
        model.text_encoder.load_state_dict(
            transformers.AutoModel.from_pretrained(model_args.text_encoder_name, trust_remote_code=True).state_dict()
        )
    return model


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (training_args.do_train and training_args.do_eval) or training_args.dataloader_num_workers > 0:
        updated_args = training_args.to_dict()
        updated_args["dataloader_pin_memory"] = True
        if training_args.do_train and training_args.do_eval:
            updated_args["load_best_model_at_end"] = True
            if updated_args.get("metric_for_best_model") is None:
                updated_args["metric_for_best_model"] = "eval_local_r@1"
            if updated_args.get("greater_is_better") is None:
                updated_args["greater_is_better"] = True
            if updated_args.get("save_strategy") != updated_args.get("eval_strategy"):
                updated_args["save_strategy"] = updated_args.get("eval_strategy")
            if str(updated_args.get("eval_strategy")).lower() in {"intervalstrategy.steps", "steps"}:
                updated_args["save_steps"] = updated_args.get("eval_steps")
        training_args = training_args.__class__(**updated_args)

    model = build_model(model_args)
    image_processor_name = model_args.vis_encoder_name or model.detector.config.vis_encoder_cfg._name_or_path

    dataset_kwargs = dict(
        anatomy_num_queries=model.num_slots,
        abnormality_num_classes=model_args.num_classes,
        image_processor_name=image_processor_name,
        image_geometry_mode=data_args.image_geometry_mode,
        region_json_root=data_args.region_json_root,
    )

    train_dataset = None
    eval_dataset = None
    if training_args.do_train:
        train_dataset = build_multi_datasets(data_args.dataset_config, stage="train", **dataset_kwargs)
    if training_args.do_eval:
        eval_split = "val" if training_args.do_train else data_args.eval_split_name
        eval_dataset = build_multi_datasets(data_args.dataset_config, stage=eval_split, **dataset_kwargs)

    data_collator = DataCollatorForTemporalGroundingDataset(
        text_encoder_name=model_args.text_encoder_name,
        text_max_length=model_args.text_max_length,
    )

    callbacks = []
    if training_args.do_train and training_args.do_eval and training_args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience,
                early_stopping_threshold=training_args.early_stopping_threshold,
            )
        )

    trainer = TemporalGroundingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(training_args.output_dir, state_dict=cpu_state_dict)

    if training_args.do_eval and not training_args.do_train:
        metric_key_prefix = "test" if data_args.eval_split_name == "test" else "eval"
        metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=metric_key_prefix)
        trainer.save_metrics(metric_key_prefix, metrics)


if __name__ == "__main__":
    train()
