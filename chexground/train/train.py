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

import pathlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import transformers
from transformers import AutoConfig

from chexground.constants import DEFAULT_TOKENS
from chexground.openmmlab_support import ensure_openmmlab_paths

ensure_openmmlab_paths()

from chexground.data.build import build_multi_datasets
from chexground.data.collator import DataCollatorForMedicalVLDataset
from chexground.libra_support import LibraConfig
from chexground.model.chexground import CheXGroundConfig, CheXGroundModel, STAGE3_REGION_TOKENS, frame_token_name
from chexground.model.temporal_grounding import GRPAConfig
from chexground.train.chexground_trainer import CheXGroundTrainer


NON_LORA_TRAINABLES_NAME = "non_lora_trainables.bin"
LORA_ADAPTER_CONFIG_NAME = "adapter_config.json"
LORA_ADAPTER_BIN_NAME = "adapter_model.bin"
LORA_ADAPTER_SAFETENSORS_NAME = "adapter_model.safetensors"


def _make_adapter_free_checkpoint_view(source: pathlib.Path, output_dir: str) -> pathlib.Path:
    view_dir = pathlib.Path(output_dir) / ".adapter_free_checkpoint_views" / f"{source.name}-{os.getpid()}"
    view_dir.mkdir(parents=True, exist_ok=True)
    excluded_names = {
        LORA_ADAPTER_CONFIG_NAME,
        LORA_ADAPTER_BIN_NAME,
        LORA_ADAPTER_SAFETENSORS_NAME,
        NON_LORA_TRAINABLES_NAME,
    }
    for item in source.iterdir():
        if item.name in excluded_names:
            continue
        link_path = view_dir / item.name
        if link_path.exists() or link_path.is_symlink():
            continue
        link_path.symlink_to(item, target_is_directory=item.is_dir())
    return view_dir


def _resolve_chexground_base_model(model: torch.nn.Module) -> torch.nn.Module:
    model = getattr(model, "module", model)
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _apply_lora_adapters(
    model: CheXGroundModel,
    training_args: transformers.TrainingArguments,
) -> torch.nn.Module:
    if not training_args.lora_enable:
        return model

    from peft import LoraConfig, get_peft_model

    chexground_model = _resolve_chexground_base_model(model)
    excluded_fragments = (
        "vision_tower",
        "mm_projector",
        "vision_resampler",
        "embed_tokens",
        "embed_in",
        "lm_head",
    )
    target_modules = []
    for name, module in chexground_model.named_modules():
        if not name.startswith("libra.model."):
            continue
        if any(fragment in name for fragment in excluded_fragments):
            continue
        if isinstance(module, torch.nn.Linear):
            target_modules.append(name)

    lora_config = LoraConfig(
        r=int(training_args.lora_r),
        lora_alpha=int(training_args.lora_alpha),
        target_modules=target_modules,
        lora_dropout=float(training_args.lora_dropout),
        bias=str(training_args.lora_bias),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    chexground_model = _resolve_chexground_base_model(model)

    for name, parameter in chexground_model.named_parameters():
        if "lora_" not in name:
            parameter.requires_grad_(False)

    train_vl_projectors = bool(getattr(training_args, "train_vl_projectors", True))
    train_tac_projector = bool(getattr(training_args, "train_tac_projector", True))
    keep_added_tokens = bool(getattr(training_args, "freeze_llm_keep_added_tokens_trainable", False))

    chexground_model.image_token_bridge.requires_grad_(train_vl_projectors)
    chexground_model.merged_roi_projector.requires_grad_(train_vl_projectors)
    chexground_model.roi_box_pos_proj.requires_grad_(train_vl_projectors)
    chexground_model.missing_roi_embedding.requires_grad_(train_vl_projectors)

    libra_model = chexground_model.get_libra().model
    libra_model.mm_projector.requires_grad_(train_tac_projector)

    if chexground_model.new_input_embs is not None:
        chexground_model.new_input_embs.requires_grad_(keep_added_tokens)
    if chexground_model.extra_lm_head is not None:
        chexground_model.extra_lm_head.requires_grad_(keep_added_tokens)
    chexground_model.enable_extra_lm_head = bool(keep_added_tokens and chexground_model.extra_lm_head is not None)

    vision_tower = chexground_model.get_vision_tower()
    vision_tower.requires_grad_(False)
    vision_tower.eval()
    grpa = chexground_model.get_grpa()
    grpa.requires_grad_(False)
    grpa.eval()
    chexground_model.get_libra().lm_head.requires_grad_(False)
    return model


def _save_lora_trainables(
    model: torch.nn.Module,
    output_dir: str,
    training_args: Optional[transformers.TrainingArguments] = None,
) -> None:
    if training_args is not None and int(training_args.local_rank) not in (-1, 0):
        return
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_to_save = getattr(model, "module", model)
    chexground_model = _resolve_chexground_base_model(model_to_save)
    chexground_model.config.save_pretrained(output_path)
    if chexground_model.generation_config is not None:
        chexground_model.generation_config.save_pretrained(output_path)

    if hasattr(model_to_save, "peft_config"):
        bias = str(getattr(training_args, "lora_bias", "none"))
        named_parameters = list(model_to_save.named_parameters())
        if bias == "none":
            selected = {name: parameter for name, parameter in named_parameters if "lora_" in name}
        elif bias == "all":
            selected = {
                name: parameter
                for name, parameter in named_parameters
                if "lora_" in name or "bias" in name
            }
        elif bias == "lora_only":
            selected = {}
            maybe_lora_bias = {}
            lora_bias_names = set()
            for name, parameter in named_parameters:
                if "lora_" in name:
                    selected[name] = parameter
                    lora_bias_names.add(name.split("lora_")[0] + "bias")
                elif "bias" in name:
                    maybe_lora_bias[name] = parameter
            selected.update(
                {
                    name: parameter
                    for name, parameter in maybe_lora_bias.items()
                    if name in lora_bias_names
                }
            )
        adapter_state = {name: parameter.detach().cpu().clone() for name, parameter in selected.items()}
        model_to_save.save_pretrained(output_path, state_dict=adapter_state)

    non_lora_trainables = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model_to_save.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    }
    torch.save(non_lora_trainables, output_path / NON_LORA_TRAINABLES_NAME)


def _load_lora_checkpoint_if_available(
    model: torch.nn.Module,
    checkpoint_dir: Optional[pathlib.Path],
    training_args: transformers.TrainingArguments,
    *,
    require_trainables: bool = False,
) -> None:
    if not training_args.lora_enable or checkpoint_dir is None:
        return

    model_to_load = getattr(model, "module", model)
    adapter_bin = checkpoint_dir / LORA_ADAPTER_BIN_NAME
    adapter_safetensors = checkpoint_dir / LORA_ADAPTER_SAFETENSORS_NAME
    if require_trainables or adapter_bin.exists() or adapter_safetensors.exists():
        from peft import set_peft_model_state_dict

        if adapter_bin.exists():
            adapter_state = torch.load(adapter_bin, map_location="cpu")
        else:
            from safetensors.torch import load_file

            adapter_state = load_file(str(adapter_safetensors), device="cpu")
        set_peft_model_state_dict(model_to_load, adapter_state)

    non_lora_path = checkpoint_dir / NON_LORA_TRAINABLES_NAME
    if require_trainables or non_lora_path.exists():
        non_lora_state = torch.load(non_lora_path, map_location="cpu")
        model_to_load.load_state_dict(non_lora_state, strict=False)


class LoraTrainablesSaveCallback(transformers.TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if not bool(getattr(args, "lora_enable", False)):
            return control
        checkpoint_dir = pathlib.Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        model = kwargs.get("model")
        if model is not None:
            _save_lora_trainables(model, str(checkpoint_dir), training_args=args)
        return control


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    libra: Optional[str] = field(default=None)
    grpa: Optional[str] = field(default=None)
    vision_tower_override: Optional[str] = field(default=None)
    reset_tac_projector: bool = field(default=False)
    max_temporal_frames: int = field(default=2)
    tokenizer_max_temporal_frames: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    dataset_config: str = field(default="chexground/data/configs/chexground_pretrain.py")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    freeze_llm: bool = field(default=True)
    freeze_llm_keep_added_tokens_trainable: bool = field(default=False)
    freeze_visual: bool = field(default=True)
    train_vl_projectors: bool = field(default=True)
    train_tac_projector: bool = field(default=True)
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=128)
    lora_alpha: int = field(default=256)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    require_lora_checkpoint_trainables: bool = field(default=False)
    lora_base_model_path: Optional[str] = field(default=None)
    save_safetensors: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=True)
    model_max_length: int = field(default=2048)
    group_by_data_source: Optional[bool] = field(default=True)


def _force_libra_attn_implementation(
    model: CheXGroundModel,
    attn_implementation: Optional[str],
) -> None:
    if not attn_implementation:
        return

    candidate_configs = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "libra_cfg", None),
        getattr(model.get_libra(), "config", None),
        getattr(getattr(model.get_libra(), "model", None), "config", None),
    ]
    seen = set()
    for cfg in candidate_configs:
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        setattr(cfg, "attn_implementation", attn_implementation)
        setattr(cfg, "_attn_implementation", attn_implementation)
        if hasattr(cfg, "_attn_implementation_autoset"):
            setattr(cfg, "_attn_implementation_autoset", False)
        if hasattr(cfg, "output_attentions"):
            setattr(cfg, "output_attentions", False)


def _build_tokenizer(tokenizer_path: str, model_max_length: int, cache_dir: Optional[str], max_temporal_frames: int):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    original_vocab_size = len(tokenizer)
    special_tokens_to_add = {}
    if tokenizer.pad_token is None:
        special_tokens_to_add["pad_token"] = DEFAULT_TOKENS["pad"]

    special_tokens_map_extended = getattr(tokenizer, "special_tokens_map_extended", {}) or {}
    registered_special_tokens = set(special_tokens_map_extended.get("additional_special_tokens", []) or [])
    tokens = [
        DEFAULT_TOKENS["image"],
        DEFAULT_TOKENS["region"],
        DEFAULT_TOKENS["bor"],
        DEFAULT_TOKENS["eor"],
        DEFAULT_TOKENS["rbox"],
        DEFAULT_TOKENS["rfeat"],
        DEFAULT_TOKENS["gbox"],
    ]
    tokens.extend(frame_token_name(frame_index) for frame_index in range(max_temporal_frames))
    tokens.extend(STAGE3_REGION_TOKENS)
    additional_special_tokens = [
        token for token in tokens if token not in registered_special_tokens
    ]
    if additional_special_tokens:
        special_tokens_to_add["additional_special_tokens"] = additional_special_tokens

    if special_tokens_to_add:
        tokenizer.add_special_tokens(special_tokens_to_add)
    new_token_ids = list(range(original_vocab_size, len(tokenizer)))
    return tokenizer, new_token_ids


def _resolve_resume_checkpoint_dir(bootstrap_mode: str, output_dir: str) -> Optional[pathlib.Path]:
    if bootstrap_mode == "compose":
        return None
    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
    checkpoint_dirs = []
    for checkpoint_dir in pathlib.Path(output_dir).glob("checkpoint-*"):
        if not checkpoint_dir.is_dir():
            continue
        match = checkpoint_pattern.match(checkpoint_dir.name)
        if match is None:
            continue
        checkpoint_dirs.append((int(match.group(1)), checkpoint_dir))
    if not checkpoint_dirs:
        return None
    return max(checkpoint_dirs, key=lambda item: item[0])[1]


def train(attn_implementation=None):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if torch.cuda.is_available() and 0 <= local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.gradient_checkpointing and training_args.gradient_checkpointing_kwargs is None:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    bootstrap_mode = "checkpoint" if model_args.model_name_or_path else "compose"
    resume_checkpoint_dir = _resolve_resume_checkpoint_dir(bootstrap_mode, training_args.output_dir)
    model_torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else None
    checkpoint_bootstrap_path = pathlib.Path(model_args.model_name_or_path) if bootstrap_mode == "checkpoint" else None
    checkpoint_has_adapter = checkpoint_bootstrap_path is not None and any(
        (checkpoint_bootstrap_path / name).exists() for name in (LORA_ADAPTER_BIN_NAME, LORA_ADAPTER_SAFETENSORS_NAME)
    )
    model_load_path = model_args.model_name_or_path
    tokenizer_checkpoint_path = model_args.model_name_or_path
    lora_trainables_checkpoint_dir = resume_checkpoint_dir
    require_lora_trainables_restore = False
    force_lora_trainables_restore = bool(getattr(training_args, "require_lora_checkpoint_trainables", False))
    explicit_lora_base_model = getattr(training_args, "lora_base_model_path", None)
    if isinstance(explicit_lora_base_model, str):
        explicit_lora_base_model = explicit_lora_base_model.strip() or None
    if (
        bootstrap_mode == "checkpoint"
        and checkpoint_bootstrap_path is not None
        and bool(getattr(training_args, "lora_enable", False))
        and resume_checkpoint_dir is None
        and (
            force_lora_trainables_restore
            or checkpoint_has_adapter
            or (checkpoint_bootstrap_path / NON_LORA_TRAINABLES_NAME).exists()
        )
    ):
        lora_trainables_checkpoint_dir = checkpoint_bootstrap_path
        require_lora_trainables_restore = True
        if checkpoint_has_adapter:
            adapter_base_model = explicit_lora_base_model
            adapter_config_path = checkpoint_bootstrap_path / LORA_ADAPTER_CONFIG_NAME
            if adapter_base_model is None and adapter_config_path.exists():
                with adapter_config_path.open("r", encoding="utf-8") as handle:
                    adapter_config = json.load(handle)
                base_model = adapter_config.get("base_model_name_or_path")
                if isinstance(base_model, str) and base_model.strip():
                    adapter_base_model = base_model.strip()
            if adapter_base_model is not None and (
                pathlib.Path(adapter_base_model).expanduser().resolve(strict=False)
                != checkpoint_bootstrap_path.expanduser().resolve(strict=False)
            ):
                model_load_path = adapter_base_model
                if not any(
                    (checkpoint_bootstrap_path / name).exists()
                    for name in ("tokenizer_config.json", "tokenizer.json", "tokenizer.model", "special_tokens_map.json", "vocab.json")
                ):
                    tokenizer_checkpoint_path = adapter_base_model
            elif any(
                (checkpoint_bootstrap_path / name).exists()
                for name in ("pytorch_model.bin", "model.safetensors", "pytorch_model.bin.index.json", "model.safetensors.index.json")
            ):
                model_load_path = str(
                    _make_adapter_free_checkpoint_view(checkpoint_bootstrap_path, training_args.output_dir)
                )

    tokenizer_token_reserve_frames = model_args.tokenizer_max_temporal_frames or model_args.max_temporal_frames

    if bootstrap_mode == "compose":
        with open(pathlib.Path(model_args.grpa) / "config.json", "r", encoding="utf-8") as handle:
            grpa_checkpoint_config = json.load(handle)

        text_encoder_name = grpa_checkpoint_config.get("text_encoder_name")
        text_encoder_override = os.getenv("CHEXGROUND_TEXT_ENCODER_OVERRIDE")

        resolved_text_encoder = None
        if isinstance(text_encoder_name, str) and pathlib.Path(text_encoder_name).exists():
            resolved_text_encoder = text_encoder_name
        elif isinstance(text_encoder_override, str) and text_encoder_override.strip() and pathlib.Path(text_encoder_override).exists():
            resolved_text_encoder = text_encoder_override

        if resolved_text_encoder is not None:
            grpa_checkpoint_config["text_encoder_name"] = resolved_text_encoder
            grpa_checkpoint_config["text_encoder_cfg"] = AutoConfig.from_pretrained(
                resolved_text_encoder,
                trust_remote_code=True,
                local_files_only=True,
            )

        grpa_cfg = GRPAConfig(**grpa_checkpoint_config)
        libra_cfg = LibraConfig.from_pretrained(
            model_args.libra, cache_dir=training_args.cache_dir, trust_remote_code=True,
        )
        shared_visual_source = model_args.vision_tower_override or libra_cfg.mm_vision_tower
        tokenizer_path = model_args.libra
    else:
        tokenizer_path = tokenizer_checkpoint_path

    tokenizer, new_token_ids = _build_tokenizer(
        tokenizer_path=tokenizer_path,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
        max_temporal_frames=tokenizer_token_reserve_frames,
    )

    if bootstrap_mode == "checkpoint":
        model_load_kwargs: Dict[str, Any] = {
            "cache_dir": training_args.cache_dir,
            "attn_implementation": attn_implementation,
        }
        if model_torch_dtype is not None:
            model_load_kwargs["torch_dtype"] = model_torch_dtype
        model = CheXGroundModel.from_pretrained(model_load_path, **model_load_kwargs)
    else:
        libra_cfg.mm_vision_tower = shared_visual_source
        model_cfg = CheXGroundConfig(
            libra_cfg=libra_cfg,
            grpa_cfg=grpa_cfg,
            max_temporal_frames=model_args.max_temporal_frames,
            num_slots=len(STAGE3_REGION_TOKENS),
            num_new_token=len(new_token_ids),
            image_processor_name=shared_visual_source,
            vision_tower_override=model_args.vision_tower_override,
        )
        model = CheXGroundModel(
            model_cfg,
            pretrained_libra=model_args.libra,
            pretrained_grpa=model_args.grpa,
            vision_tower_override=model_args.vision_tower_override,
            attn_implementation=attn_implementation,
            cache_dir=training_args.cache_dir,
            torch_dtype=model_torch_dtype,
        )

    if model_args.reset_tac_projector:
        model.reset_tac_projector()

    _force_libra_attn_implementation(model, attn_implementation)

    required_num_new_tokens = len(tokenizer) - model.base_vocab_size
    if required_num_new_tokens != int(model.config.num_new_token):
        model.resize_new_token_layers(required_num_new_tokens)
    model.init_special_token_id(tokenizer)

    vis_processor = transformers.AutoImageProcessor.from_pretrained(
        model.config.image_processor_name,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.do_sample = True

    if training_args.freeze_visual:
        model.freeze_visual()
    if training_args.freeze_llm:
        model.freeze_llm(
            keep_added_token_path_trainable=training_args.freeze_llm_keep_added_tokens_trainable,
        )
    if not training_args.train_tac_projector:
        model.get_libra().model.mm_projector.requires_grad_(False)
    if not training_args.train_vl_projectors:
        model.image_token_bridge.requires_grad_(False)
        model.merged_roi_projector.requires_grad_(False)
        model.roi_box_pos_proj.requires_grad_(False)
        model.missing_roi_embedding.requires_grad_(False)

    model = _apply_lora_adapters(model, training_args)
    _load_lora_checkpoint_if_available(
        model,
        lora_trainables_checkpoint_dir,
        training_args,
        require_trainables=require_lora_trainables_restore,
    )
    runtime_model = _resolve_chexground_base_model(model)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(training_args, model)
    optimizer = optimizer_cls(
        [{"params": trainable_params, "weight_decay": training_args.weight_decay}], **optimizer_kwargs,
    )

    train_datasets = build_multi_datasets(
        data_args.dataset_config,
        tokenizer=tokenizer,
        img_processor=vis_processor,
        stage="train",
    )
    eval_datasets = build_multi_datasets(
        data_args.dataset_config,
        tokenizer=tokenizer,
        img_processor=vis_processor,
        stage="val",
    )
    data_collator = DataCollatorForMedicalVLDataset(
        tokenizer=tokenizer,
        max_temporal_frames=runtime_model.config.max_temporal_frames,
    )

    if eval_datasets is not None:
        training_args.do_eval = True
        if getattr(training_args, "eval_steps", None):
            training_args.eval_strategy = "steps"
    callbacks = []
    if training_args.lora_enable:
        callbacks.append(LoraTrainablesSaveCallback())

    trainer = CheXGroundTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )

    if resume_checkpoint_dir is not None:
        trainer.train(resume_from_checkpoint=str(resume_checkpoint_dir))
    else:
        trainer.train()

    trainer.save_model()
    if training_args.lora_enable:
        _save_lora_trainables(trainer.model, training_args.output_dir, training_args=training_args)
    trainer.save_state()
    vis_processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
