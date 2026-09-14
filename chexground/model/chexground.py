import copy
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForCausalLM, GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from chexground.constants import DEFAULT_TOKENS, IGNORE_INDEX
from chexground.libra_support import LibraConfig, LibraLlamaForCausalLM, build_vision_tower
from chexground.model.temporal_grounding import GRPA, GRPAConfig, build_1d_sincos_pos_embed

STAGE3_REGION_TOKENS = [f"<r{i}>" for i in range(29)]


def frame_token_name(frame_index: int) -> str:
    if frame_index < 0:
        raise ValueError(f"frame_index must be non-negative, got {frame_index}.")
    if frame_index == 0:
        return DEFAULT_TOKENS["curr"]
    key = f"previm{frame_index}"
    return DEFAULT_TOKENS.get(key, f"<previm{frame_index}>")


class CheXGroundConfig(PretrainedConfig):
    model_type = "chexground"

    def __init__(
        self, libra_cfg: Optional[Union[dict, LibraConfig]] = None, grpa_cfg: Optional[Union[dict, GRPAConfig]] = None, max_temporal_frames: int = 2,
        num_slots: int = 29, num_new_token: int = 0, image_processor_name: Optional[str] = None, vision_tower_override: Optional[str] = None, **kwargs,
    ):
        super().__init__(**kwargs)
        if libra_cfg is None:
            self.libra_cfg = LibraConfig()
        elif isinstance(libra_cfg, dict):
            libra_payload = dict(libra_cfg)
            libra_payload.pop("model_type", None)
            self.libra_cfg = LibraConfig(**libra_payload)
        elif isinstance(libra_cfg, LibraConfig):
            self.libra_cfg = libra_cfg
        else:
            raise TypeError(f"Unsupported libra_cfg type: {type(libra_cfg)!r}")

        if grpa_cfg is None:
            self.grpa_cfg = GRPAConfig(text_encoder_name=None)
        elif isinstance(grpa_cfg, dict):
            grpa_payload = dict(grpa_cfg)
            grpa_payload.pop("model_type", None)
            self.grpa_cfg = GRPAConfig(**grpa_payload)
        elif isinstance(grpa_cfg, GRPAConfig):
            self.grpa_cfg = grpa_cfg
        else:
            raise TypeError(f"Unsupported grpa_cfg type: {type(grpa_cfg)!r}")

        if max_temporal_frames <= 0:
            raise ValueError(f"max_temporal_frames must be positive, got {max_temporal_frames}.")

        self.max_temporal_frames = int(max_temporal_frames)
        self.num_slots = int(num_slots)
        self.num_new_token = int(num_new_token)
        self.image_processor_name = image_processor_name
        self.vision_tower_override = vision_tower_override
        if self.vision_tower_override:
            self.libra_cfg.mm_vision_tower = self.vision_tower_override
        self.vocab_size = int(getattr(self.libra_cfg, "vocab_size", 0)) + int(self.num_new_token)
        self.hidden_size = int(getattr(self.libra_cfg, "hidden_size", 0))
        self.bos_token_id = self.libra_cfg.bos_token_id
        self.eos_token_id = self.libra_cfg.eos_token_id
        self.pad_token_id = self.libra_cfg.pad_token_id
        self.output_attentions = self.libra_cfg.output_attentions
        self.output_hidden_states = self.libra_cfg.output_hidden_states
        self.tie_word_embeddings = self.libra_cfg.tie_word_embeddings
        self.return_dict = bool(self.libra_cfg.return_dict)
        self.mm_projector_type = getattr(self.libra_cfg, "mm_projector_type", None)
        self.mm_hidden_size = getattr(self.libra_cfg, "mm_hidden_size", None)
        self.mm_vision_tower = getattr(self.libra_cfg, "mm_vision_tower", None)

    def to_json_string(self, use_diff: bool = True) -> str:
        config_dict = self.to_diff_dict() if use_diff else self.to_dict()
        config_dict["libra_cfg"] = json.loads(self.libra_cfg.to_json_string(use_diff=use_diff))
        config_dict["grpa_cfg"] = json.loads(self.grpa_cfg.to_json_string(use_diff=use_diff))
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"


class CheXGroundModel(PreTrainedModel, GenerationMixin):
    config_class = CheXGroundConfig
    _keys_to_ignore_on_load_unexpected = [r"roi_projector\..*"]
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(
        self, config: CheXGroundConfig, pretrained_libra: Optional[str] = None, pretrained_grpa: Optional[str] = None,
        vision_tower_override: Optional[str] = None, attn_implementation: Optional[str] = None, cache_dir: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None, **kwargs,
    ):
        super().__init__(config)
        del kwargs
        if bool(pretrained_libra) != bool(pretrained_grpa):
            raise ValueError("pretrained_libra and pretrained_grpa must be provided together.")
        if pretrained_libra is None and vision_tower_override is not None:
            raise ValueError("vision_tower_override is only supported during fresh Libra + GRPA composition.")

        if attn_implementation is not None:
            setattr(config.libra_cfg, "attn_implementation", attn_implementation)
            setattr(config.libra_cfg, "_attn_implementation", attn_implementation)
        if pretrained_libra is not None:
            libra_load_kwargs: Dict[str, Any] = {"config": config.libra_cfg, "cache_dir": cache_dir}
            if attn_implementation is not None:
                libra_load_kwargs["attn_implementation"] = attn_implementation
            if torch_dtype is not None:
                libra_load_kwargs["torch_dtype"] = torch_dtype
            self.libra = LibraLlamaForCausalLM.from_pretrained(pretrained_libra, **libra_load_kwargs)

            grpa_load_kwargs: Dict[str, Any] = {}
            if config.grpa_cfg is not None:
                grpa_load_kwargs["config"] = config.grpa_cfg
            if torch_dtype is not None:
                grpa_load_kwargs["torch_dtype"] = torch_dtype
            self.grpa = GRPA.from_pretrained(pretrained_grpa, **grpa_load_kwargs)
        else:
            self.libra = LibraLlamaForCausalLM(config.libra_cfg)
            self.grpa = GRPA(config.grpa_cfg)

        self._sync_component_configs()
        resolved_vision_source = vision_tower_override or config.vision_tower_override or self.libra.config.mm_vision_tower
        self.libra.config.mm_vision_tower = resolved_vision_source
        self.libra.model.vision_tower = build_vision_tower(self.libra.config, delay_load=pretrained_libra is None)
        if pretrained_libra is None:
            vision_tower = self.libra.model.vision_tower
            if not getattr(vision_tower, "is_loaded", False):
                if not hasattr(vision_tower, "vision_tower"):
                    vision_tower_name = vision_tower.vision_tower_name
                    cfg_only = getattr(vision_tower, "cfg_only", None)
                    if cfg_only is None:
                        cfg_only = AutoConfig.from_pretrained(vision_tower_name, trust_remote_code=True)
                        vision_tower.cfg_only = cfg_only
                    vision_tower.image_processor = AutoImageProcessor.from_pretrained(vision_tower_name, trust_remote_code=True)
                    vision_tower.vision_tower = AutoModel.from_config(cfg_only, trust_remote_code=True)
                    vision_tower.vision_tower.requires_grad_(False)
                vision_tower.is_loaded = True
            self.libra.model.vision_tower = vision_tower
        self.libra.config.mm_hidden_size = int(self.get_vision_tower().hidden_size)
        if getattr(self.libra.config, "mm_projector_type", None) != "TAC":
            raise ValueError(f"Expected Libra TAC projector, got {self.libra.config.mm_projector_type!r}.")

        self.text_hidden_size = int(self.libra.config.hidden_size)
        self.vision_hidden_size = int(getattr(self.libra.config, "mm_hidden_size", self.get_vision_tower().hidden_size))
        self.visual_hidden_size = int(self.grpa.visual_dim)
        self.base_vocab_size = int(self.libra.config.vocab_size)
        self.num_slots = min(int(config.num_slots), int(self.grpa.num_slots))
        self.config.num_slots = self.num_slots

        self.prior_token_compressor = nn.Linear(self.text_hidden_size, self.vision_hidden_size)
        self.image_token_bridge = self._build_projector(self.vision_hidden_size * 4, self.text_hidden_size)
        self.merged_roi_projector = self._build_projector(self.visual_hidden_size * int(self.config.max_temporal_frames), self.text_hidden_size)
        self.roi_box_pos_proj = nn.Sequential(nn.Linear(4, 256), nn.ReLU(inplace=True), nn.LayerNorm(256), nn.Linear(256, self.visual_hidden_size),
            nn.ReLU(inplace=True), nn.LayerNorm(self.visual_hidden_size))
        self.missing_roi_embedding = nn.Parameter(torch.zeros(self.visual_hidden_size))
        self.extra_lm_head = (nn.Linear(self.text_hidden_size, config.num_new_token, bias=False) if config.num_new_token > 0 else None)
        self.enable_extra_lm_head = self.extra_lm_head is not None
        self.new_input_embs = (nn.Embedding(config.num_new_token, self.text_hidden_size) if config.num_new_token > 0 else None)
        if self.new_input_embs is not None:
            input_embeds = self.libra.get_input_embeddings().weight.data
            input_embeds_avg = input_embeds.mean(dim=0, keepdim=True)
            self.new_input_embs.weight.data[:, :] = input_embeds_avg

        self.pad_token_id: Optional[int] = None
        self.image_token_id: Optional[int] = None
        self.region_token_id: Optional[int] = None
        self.refer_box_token_id: Optional[int] = None
        self.refer_feat_token_id: Optional[int] = None
        self.ground_box_token_id: Optional[int] = None
        self.frame_token_ids: Dict[str, int] = {}
        self.region_index_token_ids: List[int] = []
        self.box_idx_token_ids: List[int] = []
        self.accepts_loss_kwargs = False

        self._sync_component_configs()
        self.config.image_processor_name = resolved_vision_source
        self.config.libra_cfg.mm_vision_tower = self.config.vision_tower_override or resolved_vision_source
        self.config.mm_hidden_size = int(self.get_vision_tower().hidden_size)
        self.base_vocab_size = int(self.libra.config.vocab_size)
        self.config.vocab_size = self.base_vocab_size + int(self.config.num_new_token)
        self.config.hidden_size = int(self.libra.config.hidden_size)
        self.config.num_slots = self.num_slots
        self._init_new_parameters()
        self.generation_config = copy.deepcopy(self.libra.generation_config)

    @staticmethod
    def _build_projector(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim))

    def _init_new_parameters(self) -> None:
        for module in [self.prior_token_compressor, self.image_token_bridge, self.merged_roi_projector]:
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
        nn.init.normal_(self.missing_roi_embedding, std=0.02)

    def _sync_component_configs(self) -> None:
        self.config.libra_cfg = self.libra.config
        self.config.grpa_cfg = self.grpa.config
        config = self.config
        if config.vision_tower_override:
            config.libra_cfg.mm_vision_tower = config.vision_tower_override
        config.vocab_size = int(getattr(config.libra_cfg, "vocab_size", 0)) + int(config.num_new_token)
        config.hidden_size = int(getattr(config.libra_cfg, "hidden_size", 0))
        config.bos_token_id = config.libra_cfg.bos_token_id
        config.eos_token_id = config.libra_cfg.eos_token_id
        config.pad_token_id = config.libra_cfg.pad_token_id
        config.output_attentions = config.libra_cfg.output_attentions
        config.output_hidden_states = config.libra_cfg.output_hidden_states
        config.tie_word_embeddings = config.libra_cfg.tie_word_embeddings
        config.return_dict = bool(config.libra_cfg.return_dict)
        config.mm_projector_type = getattr(config.libra_cfg, "mm_projector_type", None)
        config.mm_hidden_size = getattr(config.libra_cfg, "mm_hidden_size", None)
        config.mm_vision_tower = getattr(config.libra_cfg, "mm_vision_tower", None)

    @property
    def model(self):
        return self.libra.model

    @property
    def lm_head(self):
        return self.libra.lm_head

    def get_model(self):
        return self.libra.get_model()

    def init_special_token_id(self, tokenizer) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        self.image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_TOKENS["image"])
        self.region_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_TOKENS["region"])
        self.refer_box_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_TOKENS["rbox"])
        self.refer_feat_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_TOKENS["rfeat"])
        self.ground_box_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_TOKENS["gbox"])
        self.frame_token_ids = {frame_token_name(frame_index): tokenizer.convert_tokens_to_ids(frame_token_name(frame_index))
            for frame_index in range(int(self.config.max_temporal_frames))}
        self.region_index_token_ids = tokenizer.convert_tokens_to_ids(STAGE3_REGION_TOKENS)
        self.box_idx_token_ids = list(self.region_index_token_ids)

    def resize_new_token_layers(self, num_new_tokens: int) -> None:
        num_new_tokens = int(num_new_tokens)
        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens must be non-negative, got {num_new_tokens}.")

        current_num_new_tokens = int(getattr(self.config, "num_new_token", 0))
        if num_new_tokens == current_num_new_tokens:
            return
        if num_new_tokens < current_num_new_tokens:
            raise ValueError("Shrinking CheXGround added-token tables is not supported. " f"current={current_num_new_tokens} requested={num_new_tokens}.")

        input_embed_weight = self.libra.get_input_embeddings().weight.data
        input_embed_avg = input_embed_weight.mean(dim=0, keepdim=True)
        output_embed_avg = self.libra.lm_head.weight.data.mean(dim=0, keepdim=True)
        input_embed_device = input_embed_weight.device
        input_embed_dtype = input_embed_weight.dtype
        output_embed_device = self.libra.lm_head.weight.device
        output_embed_dtype = self.libra.lm_head.weight.dtype

        old_new_input_embs = self.new_input_embs
        new_input_embs = nn.Embedding(num_new_tokens, self.text_hidden_size).to(device=input_embed_device, dtype=input_embed_dtype)
        new_input_embs.weight.data[:, :] = input_embed_avg.to(dtype=new_input_embs.weight.dtype)
        if old_new_input_embs is not None and current_num_new_tokens > 0:
            preserved = min(current_num_new_tokens, num_new_tokens)
            new_input_embs.weight.data[:preserved] = old_new_input_embs.weight.data[:preserved].to(device=new_input_embs.weight.device,
                dtype=new_input_embs.weight.dtype)
        self.new_input_embs = new_input_embs

        old_extra_lm_head = self.extra_lm_head
        new_extra_lm_head = nn.Linear(self.text_hidden_size, num_new_tokens, bias=False).to(device=output_embed_device, dtype=output_embed_dtype)
        new_extra_lm_head.weight.data[:, :] = output_embed_avg.to(dtype=new_extra_lm_head.weight.dtype)
        if old_extra_lm_head is not None and current_num_new_tokens > 0:
            preserved = min(current_num_new_tokens, num_new_tokens)
            new_extra_lm_head.weight.data[:preserved] = old_extra_lm_head.weight.data[:preserved].to(device=new_extra_lm_head.weight.device,
                dtype=new_extra_lm_head.weight.dtype)
        self.extra_lm_head = new_extra_lm_head
        if num_new_tokens > 0 and old_extra_lm_head is None:
            self.enable_extra_lm_head = True

        self.config.num_new_token = num_new_tokens
        self.config.vocab_size = self.base_vocab_size + num_new_tokens

    def get_grpa(self) -> GRPA:
        return self.grpa

    def get_libra(self) -> LibraLlamaForCausalLM:
        return self.libra

    def get_vision_tower(self):
        return self.libra.get_vision_tower()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.libra.encode_images(images)

    def _extract_shared_backbone_outputs(self, images: torch.Tensor, temporal_lengths: torch.LongTensor) -> Dict[str, torch.Tensor]:
        batch_size, total_frames = images.shape[:2]
        temporal_lengths = temporal_lengths.to(device=images.device, dtype=torch.long)
        time_index = torch.arange(total_frames, device=images.device, dtype=torch.long).view(1, total_frames)
        first_valid_index = total_frames - temporal_lengths.view(batch_size, 1)
        valid_frame_mask = time_index >= first_valid_index
        flat_images = images.reshape(batch_size * total_frames, *images.shape[2:])
        flat_valid_mask = valid_frame_mask.reshape(-1)
        valid_images = flat_images[flat_valid_mask]

        vision_backbone = self.get_vision_tower().vision_tower
        target_param = next(vision_backbone.parameters())
        if valid_images.device != target_param.device or valid_images.dtype != target_param.dtype:
            valid_images = valid_images.to(device=target_param.device, dtype=target_param.dtype)

        image_forward_outs = vision_backbone(valid_images, output_hidden_states=True)
        return {
            "image_forward_outs": image_forward_outs,
            "batch_size": batch_size,
            "total_frames": total_frames,
            "valid_frame_mask": valid_frame_mask,
        }

    def get_input_embeddings(self):
        return self.libra.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.libra.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.libra.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.libra.set_output_embeddings(value)

    def build_input_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        embed_tokens = self.libra.get_input_embeddings()
        if self.new_input_embs is None:
            return embed_tokens(input_ids)

        mask = input_ids >= self.base_vocab_size
        base_input_ids = input_ids.masked_fill(mask, 0)
        new_input_ids = (input_ids - self.base_vocab_size).masked_fill(~mask, 0)

        input_embeddings = embed_tokens(base_input_ids)
        new_input_embeddings = self.new_input_embs(new_input_ids).to(device=input_embeddings.device, dtype=input_embeddings.dtype)
        input_embeddings[mask] = new_input_embeddings[mask]
        return input_embeddings

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.libra.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.libra.gradient_checkpointing_disable()

    def reset_tac_projector(self) -> None:
        if getattr(self.libra.config, "mm_projector_type", None) != "TAC":
            raise ValueError(f"Expected Libra TAC projector, got {self.libra.config.mm_projector_type!r}.")
        from libra.model.multimodal_projector.builder import build_vision_projector

        self.libra.model.mm_projector = build_vision_projector(self.libra.config)

    def freeze_llm(self, keep_added_token_path_trainable: bool = False) -> None:
        for name, param in self.libra.model.named_parameters():
            if name.startswith("vision_tower") or name.startswith("mm_projector"):
                continue
            param.requires_grad_(False)
        self.libra.lm_head.requires_grad_(False)
        if self.new_input_embs is not None and keep_added_token_path_trainable:
            self.new_input_embs.requires_grad_(True)
        if self.extra_lm_head is not None:
            self.enable_extra_lm_head = bool(keep_added_token_path_trainable)
            self.extra_lm_head.requires_grad_(bool(keep_added_token_path_trainable))

    def freeze_visual(self) -> None:
        vision_tower = self.get_vision_tower()
        if vision_tower is not None:
            vision_tower.requires_grad_(False)
            vision_tower.eval()
        self.grpa.requires_grad_(False)
        self.grpa.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        vision_tower = self.get_vision_tower()
        if vision_tower is not None and not any(param.requires_grad for param in vision_tower.parameters()):
            vision_tower.eval()
        if not any(param.requires_grad for param in self.grpa.parameters()):
            self.grpa.eval()
        return self

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values is not None and input_ids is not None:
            input_ids = input_ids[:, -1:]
        model_inputs = {"inputs_embeds": inputs_embeds} if (inputs_embeds is not None and past_key_values is None) else {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "use_cache": kwargs.get("use_cache"),
                "images": kwargs.get("images"),
                "pixel_mask": kwargs.get("pixel_mask"),
                "temporal_lengths": kwargs.get("temporal_lengths"),
                "refer_slot_ids": kwargs.get("refer_slot_ids"),
                "ground_slot_ids": kwargs.get("ground_slot_ids"),
            }
        )
        return model_inputs

    def _extract_tac_tokens(self, shared_backbone_outputs: Dict[str, torch.Tensor], temporal_lengths: torch.LongTensor) -> Dict[str, torch.Tensor]:
        vision_tower = self.get_vision_tower()
        if (getattr(vision_tower, "select_layer", "all"), getattr(vision_tower, "select_feature", "patch")) != ("all", "patch"):
            raise ValueError("CheXGround TAC requires all-layer patch features.")

        # Scatter all-layer patch features back into the left-padded frame layout.
        hidden_states = shared_backbone_outputs["image_forward_outs"].hidden_states
        tac_valid_features = torch.stack([hidden_state[:, 1:, :].contiguous() for hidden_state in hidden_states[1:]], dim=1)
        batch_size = int(shared_backbone_outputs["batch_size"])
        total_frames = int(shared_backbone_outputs["total_frames"])
        valid_frame_mask = shared_backbone_outputs["valid_frame_mask"]
        tac_frame_features = tac_valid_features.new_zeros((batch_size * total_frames, *tac_valid_features.shape[1:]))
        tac_frame_features[valid_frame_mask.reshape(-1).to(device=tac_valid_features.device)] = tac_valid_features
        tac_frame_features = tac_frame_features.reshape(batch_size, total_frames, *tac_valid_features.shape[1:])

        # Each frame attends to its predecessor; the first valid frame pairs with itself.
        batch_size, temporal_length = tac_frame_features.shape[:2]
        temporal_lengths = temporal_lengths.to(device=tac_frame_features.device, dtype=torch.long)
        time_index = torch.arange(temporal_length, device=tac_frame_features.device, dtype=torch.long).view(1, temporal_length)
        first_valid_index = temporal_length - temporal_lengths.view(batch_size, 1)
        previous_index = torch.clamp(time_index - 1, min=0)
        previous_index = torch.where(time_index == first_valid_index, time_index, previous_index)
        previous_index = torch.where(time_index >= first_valid_index, previous_index, time_index)
        gather_index = previous_index.view(batch_size, temporal_length, *([1] * (tac_frame_features.ndim - 2))).expand(-1, -1, *tac_frame_features.shape[2:])
        previous = torch.gather(tac_frame_features, dim=1, index=gather_index)
        current = tac_frame_features.reshape(batch_size * temporal_length, *tac_frame_features.shape[2:])
        previous = previous.reshape(batch_size * temporal_length, *tac_frame_features.shape[2:])
        paired_features = torch.stack([current, previous], dim=0)
        flat_valid_mask = shared_backbone_outputs["valid_frame_mask"].reshape(-1).to(device=paired_features.device)
        valid_paired_features = paired_features[:, flat_valid_mask]
        valid_tac_tokens = self.get_model().mm_projector(valid_paired_features)
        tac_tokens = valid_tac_tokens.new_zeros((batch_size * temporal_length, valid_tac_tokens.shape[1], valid_tac_tokens.shape[2]))
        tac_tokens[flat_valid_mask] = valid_tac_tokens
        full_res_tokens = tac_tokens.reshape(batch_size, temporal_length, tac_tokens.shape[1], tac_tokens.shape[2])
        compressed_tokens = self.prior_token_compressor(full_res_tokens)

        # Crop odd grid edges, then merge patches in the trained column-first order.
        batch_size, temporal_length, sequence_length, hidden_dim = compressed_tokens.shape
        height = width = int(math.sqrt(sequence_length))
        flat_tokens = compressed_tokens.reshape(batch_size * temporal_length, height, width, hidden_dim)
        even_height = height - (height % 2)
        even_width = width - (width % 2)
        flat_tokens = flat_tokens[:, :even_height, :even_width]
        merged_tokens = torch.cat([flat_tokens[:, 0::2, 0::2, :], flat_tokens[:, 1::2, 0::2, :], flat_tokens[:, 0::2, 1::2, :], flat_tokens[:, 1::2, 1::2, :],
            ], dim=-1)
        downsampled_tokens = merged_tokens.reshape(batch_size, temporal_length, -1, hidden_dim * 4)
        downsampled_tokens = self.image_token_bridge(downsampled_tokens)
        return {"full_res_tokens": full_res_tokens, "downsampled_tokens": downsampled_tokens,}

    def _build_visual_feature_lists(
        self, temporal_lengths: torch.Tensor, image_patch_tokens: Dict[str, torch.Tensor], roi_outputs: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, Any]]]:
        full_res_image_tokens = image_patch_tokens["full_res_tokens"]
        downsampled_image_tokens = image_patch_tokens["downsampled_tokens"]
        temporal_lengths = temporal_lengths.to(device=full_res_image_tokens.device, dtype=torch.long)
        roi_valid_mask = roi_outputs["temporal_roi_valid_mask"][:, :, : self.num_slots]
        temporal_roi_boxes = roi_outputs["temporal_roi_boxes"][:, :, : self.num_slots]
        temporal_roi_tokens = roi_outputs["temporal_roi_last_hidden_state"][:, :, : self.num_slots].mean(dim=3)

        flat_roi_boxes = temporal_roi_boxes.reshape(-1, temporal_roi_boxes.shape[-1])
        temporal_roi_box_pos = self.roi_box_pos_proj(flat_roi_boxes).reshape(temporal_roi_boxes.shape[0], temporal_roi_boxes.shape[1],
            temporal_roi_boxes.shape[2], self.visual_hidden_size)
        temporal_roi_box_pos = temporal_roi_box_pos.masked_fill(~roi_valid_mask[..., None], 0.0)

        temporal_roi_tokens = torch.where(roi_valid_mask[..., None], temporal_roi_tokens, self.missing_roi_embedding.view(1, 1, 1, -1))

        max_temporal_frames = int(self.config.max_temporal_frames)
        image_temporal_time_embeddings = build_1d_sincos_pos_embed(max_temporal_frames, self.text_hidden_size, device=full_res_image_tokens.device,
            dtype=full_res_image_tokens.dtype)
        roi_temporal_time_embeddings = build_1d_sincos_pos_embed(max_temporal_frames, self.visual_hidden_size, device=temporal_roi_tokens.device,
            dtype=temporal_roi_tokens.dtype)

        image_feature_list: List[torch.Tensor] = []
        roi_feature_list: List[torch.Tensor] = []
        sample_meta: List[Dict[str, Any]] = []
        fused_roi_token_list: List[torch.Tensor] = []
        total_frames = int(full_res_image_tokens.shape[1])

        for batch_idx, temporal_length in enumerate(temporal_lengths.tolist()):
            frame_specs = [{"name": frame_token_name(frame_index), "frame_embed_index": frame_index, "source_time_index": total_frames - 1 - frame_index,}
                for frame_index in range(temporal_length)]
            frame_embed_indices = torch.tensor([frame_spec["frame_embed_index"] for frame_spec in frame_specs], device=full_res_image_tokens.device,
                dtype=torch.long)
            frame_time_embeds = image_temporal_time_embeddings[frame_embed_indices]

            framewise_image_tokens: List[torch.Tensor] = []
            image_token_counts: List[int] = []
            for logical_frame_index, frame_spec in enumerate(frame_specs):
                source_time_index = int(frame_spec["source_time_index"])
                if logical_frame_index == 0:
                    frame_image_tokens = full_res_image_tokens[batch_idx, source_time_index]
                else:
                    frame_image_tokens = downsampled_image_tokens[batch_idx, source_time_index]
                frame_image_tokens = frame_image_tokens + frame_time_embeds[logical_frame_index].to(dtype=frame_image_tokens.dtype).view(1, -1)
                framewise_image_tokens.append(frame_image_tokens)
                image_token_counts.append(int(frame_image_tokens.shape[0]))

            sample_image_tokens = torch.cat(framewise_image_tokens, dim=0)

            fused_roi_components = []
            for logical_frame_index in range(max_temporal_frames):
                if logical_frame_index < len(frame_specs):
                    source_time_index = frame_specs[logical_frame_index]["source_time_index"]
                    frame_roi = temporal_roi_tokens[batch_idx, source_time_index]  # [num_slots, visual_dim]
                    frame_roi = frame_roi + temporal_roi_box_pos[batch_idx, source_time_index].to(dtype=frame_roi.dtype)
                else:
                    frame_roi = self.missing_roi_embedding.unsqueeze(0).expand(self.num_slots, -1)

                time_emb = roi_temporal_time_embeddings[logical_frame_index].to(dtype=frame_roi.dtype)
                frame_roi = frame_roi + time_emb.unsqueeze(0)
                fused_roi_components.append(frame_roi)

            fused_roi_tokens = torch.cat(fused_roi_components, dim=-1)
            fused_roi_token_list.append(fused_roi_tokens)

            sample_boxes = {frame_spec["name"]: temporal_roi_boxes[batch_idx, frame_spec["source_time_index"]] for frame_spec in frame_specs}
            sample_box_masks = {frame_spec["name"]: roi_valid_mask[batch_idx, frame_spec["source_time_index"]] for frame_spec in frame_specs}

            image_feature_list.append(sample_image_tokens.reshape(-1, sample_image_tokens.shape[-1]))
            sample_meta.append(
                {
                    "frame_specs": frame_specs,
                    "frame_boxes": sample_boxes,
                    "frame_box_masks": sample_box_masks,
                    "num_image_tokens_per_frame": image_token_counts,
                }
            )

        batched_fused_roi_tokens = torch.stack(fused_roi_token_list, dim=0)
        batched_fused_roi_tokens = self.merged_roi_projector(batched_fused_roi_tokens)
        for batch_idx, fused_roi_tokens in enumerate(batched_fused_roi_tokens.unbind(dim=0)):
            slot_permutation = list(range(self.num_slots))
            slot_inverse_permutation = list(range(self.num_slots))
            if self.training:
                rand_indices = torch.rand(self.num_slots, device=fused_roi_tokens.device).argsort()
                fused_roi_tokens = fused_roi_tokens[rand_indices]
                slot_permutation = rand_indices.tolist()
                inv_rand_indices = torch.argsort(rand_indices)
                slot_inverse_permutation = inv_rand_indices.tolist()

            roi_feature_list.append(fused_roi_tokens)
            sample_meta[batch_idx]["slot_permutation"] = slot_permutation
            sample_meta[batch_idx]["slot_inverse_permutation"] = slot_inverse_permutation

        return image_feature_list, roi_feature_list, sample_meta

    def _prepare_multimodal_inputs(
        self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor], attention_mask: torch.Tensor, temporal_lengths: torch.Tensor,
        image_patch_tokens: torch.Tensor, roi_outputs: Dict[str, torch.Tensor], refer_slot_ids: Optional[List[List[int]]] = None,
        ground_slot_ids: Optional[List[List[int]]] = None,
    ) -> Tuple[torch.LongTensor, Optional[torch.LongTensor], torch.Tensor, torch.Tensor]:
        image_feature_list, roi_feature_list, sample_meta = self._build_visual_feature_lists(temporal_lengths=temporal_lengths,
            image_patch_tokens=image_patch_tokens, roi_outputs=roi_outputs)

        expanded_input_ids = []
        expanded_labels = [] if labels is not None else None
        refer_feature_list: List[torch.Tensor] = []
        for batch_idx in range(input_ids.shape[0]):
            valid_length = int(attention_mask[batch_idx].sum().item())
            sample_input_ids = input_ids[batch_idx, :valid_length]
            sample_labels = labels[batch_idx, :valid_length] if labels is not None else None
            image_token_counts = list(sample_meta[batch_idx]["num_image_tokens_per_frame"])

            # Expand each image placeholder before rewriting grounding references.
            id_chunks = []
            label_chunks = [] if sample_labels is not None else None
            image_placeholder_index = 0
            for token_index, token_id in enumerate(sample_input_ids.tolist()):
                if token_id == self.image_token_id:
                    image_token_count = image_token_counts[image_placeholder_index]
                    image_placeholder_index += 1
                    id_chunks.append(sample_input_ids.new_full((image_token_count,), self.image_token_id))
                    if label_chunks is not None:
                        label_chunks.append(sample_labels.new_full((image_token_count,), IGNORE_INDEX))
                else:
                    id_chunks.append(sample_input_ids[token_index : token_index + 1])
                    if label_chunks is not None:
                        label_chunks.append(sample_labels[token_index : token_index + 1])
            if image_placeholder_index != len(image_token_counts):
                raise ValueError("<image> placeholder count does not match the logical frames.")
            sample_input_ids = torch.cat(id_chunks, dim=0)
            sample_labels = torch.cat(label_chunks, dim=0) if label_chunks is not None else None

            sample_input_ids = sample_input_ids.clone()
            sample_labels = sample_labels.clone() if sample_labels is not None else None
            shuffled_roi_features = roi_feature_list[batch_idx]
            slot_inverse_permutation = list(sample_meta[batch_idx]["slot_inverse_permutation"])
            sample_refer_slots = refer_slot_ids[batch_idx] if refer_slot_ids is not None else None
            sample_ground_slots = ground_slot_ids[batch_idx] if ground_slot_ids is not None else None
            sample_refer_slots = [int(slot_id) for slot_id in (sample_refer_slots or [])]
            sample_ground_slots = [int(slot_id) for slot_id in (sample_ground_slots or [])]
            refer_box_positions = torch.nonzero(sample_input_ids == self.refer_box_token_id, as_tuple=False).flatten()
            refer_feat_positions = torch.nonzero(sample_input_ids == self.refer_feat_token_id, as_tuple=False).flatten()
            ground_box_positions = torch.nonzero(sample_input_ids == self.ground_box_token_id, as_tuple=False).flatten()

            refer_region_features = shuffled_roi_features.new_empty((0, shuffled_roi_features.shape[-1]))
            if sample_refer_slots:
                refer_shuffled_positions = [int(slot_inverse_permutation[slot_id]) for slot_id in sample_refer_slots]
                refer_token_ids = sample_input_ids.new_tensor([self.region_index_token_ids[position] for position in refer_shuffled_positions],
                    dtype=torch.long)
                sample_input_ids[refer_box_positions] = refer_token_ids
                refer_region_features = shuffled_roi_features[torch.tensor(refer_shuffled_positions, device=shuffled_roi_features.device, dtype=torch.long)]

            if sample_ground_slots:
                ground_shuffled_positions = [int(slot_inverse_permutation[slot_id]) for slot_id in sample_ground_slots]
                ground_token_ids = sample_input_ids.new_tensor([self.region_index_token_ids[position] for position in ground_shuffled_positions],
                    dtype=torch.long)
                sample_input_ids[ground_box_positions] = ground_token_ids
                if sample_labels is not None:
                    supervised_ground_mask = sample_labels[ground_box_positions] != IGNORE_INDEX
                    if supervised_ground_mask.any():
                        sample_labels[ground_box_positions[supervised_ground_mask]] = ground_token_ids[supervised_ground_mask
                            ].to(device=sample_labels.device, dtype=sample_labels.dtype)

            expanded_input_ids.append(sample_input_ids)
            refer_feature_list.append(refer_region_features)
            if expanded_labels is not None:
                expanded_labels.append(sample_labels)

        input_ids = torch.nn.utils.rnn.pad_sequence(expanded_input_ids, batch_first=True, padding_value=self.pad_token_id)
        if expanded_labels is not None:
            labels = torch.nn.utils.rnn.pad_sequence(expanded_labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.pad_token_id)

        inputs_embeds = self.build_input_embeddings(input_ids)
        image_features = torch.cat(image_feature_list, dim=0).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        roi_features = torch.cat(roi_feature_list, dim=0).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        refer_features = torch.cat(refer_feature_list, dim=0).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        image_mask = input_ids == self.image_token_id
        region_mask = input_ids == self.region_token_id
        refer_feat_mask = input_ids == self.refer_feat_token_id

        inputs_embeds[image_mask] = image_features
        inputs_embeds[region_mask] = roi_features
        if refer_features.shape[0] > 0:
            inputs_embeds[refer_feat_mask] = refer_features

        return input_ids, labels, attention_mask, inputs_embeds

    def forward(
        self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, images: Optional[torch.Tensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None, temporal_lengths: Optional[torch.LongTensor] = None, refer_slot_ids: Optional[List[List[int]]] = None,
        ground_slot_ids: Optional[List[List[int]]] = None, return_dict: Optional[bool] = None, **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        _ = kwargs
        if inputs_embeds is None and past_key_values is None:
            if input_ids is None or attention_mask is None or images is None or pixel_mask is None or temporal_lengths is None:
                raise ValueError("CheXGround forward requires input_ids, attention_mask, images, pixel_mask, and temporal_lengths.")
            shared_backbone_outputs = self._extract_shared_backbone_outputs(images=images, temporal_lengths=temporal_lengths)
            image_patch_tokens = self._extract_tac_tokens(shared_backbone_outputs=shared_backbone_outputs, temporal_lengths=temporal_lengths)
            roi_outputs = self.grpa.extract_roi_features(images=images, pixel_mask=pixel_mask, temporal_lengths=temporal_lengths,
                backbone_outputs=shared_backbone_outputs)
            input_ids, labels, attention_mask, inputs_embeds = self._prepare_multimodal_inputs(input_ids=input_ids, labels=labels,
                attention_mask=attention_mask, temporal_lengths=temporal_lengths, image_patch_tokens=image_patch_tokens, roi_outputs=roi_outputs,
                refer_slot_ids=refer_slot_ids, ground_slot_ids=ground_slot_ids)
        elif past_key_values is not None and attention_mask is not None:
            token_length = past_key_values[0][0].shape[-2] + 1
            attention_mask = torch.ones((attention_mask.shape[0], token_length), device=attention_mask.device)
            if input_ids is not None:
                input_ids = input_ids[:, -1:]

        if inputs_embeds is None and input_ids is None:
            raise ValueError("Either inputs_embeds or input_ids must be provided.")
        if inputs_embeds is None:
            inputs_embeds = self.build_input_embeddings(input_ids)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.libra.model(attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict, inputs_embeds=inputs_embeds)

        hidden_states = outputs[0]
        logits = self.libra.lm_head(hidden_states)
        if self.extra_lm_head is not None and self.enable_extra_lm_head:
            extra_logits = self.extra_lm_head(hidden_states)
            logits = torch.cat((logits, extra_logits), dim=-1)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            valid_label_mask = shift_labels != IGNORE_INDEX
            if not valid_label_mask.any():
                raise ValueError("No supervised labels remain after shifting the causal LM targets.")
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)


AutoConfig.register(CheXGroundConfig.model_type, CheXGroundConfig)
AutoModelForCausalLM.register(CheXGroundConfig, CheXGroundModel)
