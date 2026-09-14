

import json
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from gloria import (GLoRIAROILocalMatcher, gloria_attention_kl_loss, gloria_positive_cosine_loss, image_to_text_contrastive_loss)
from chexground.model.ddetr import CustomDDETRConfig, CustomDDETRModel, LayerNorm
from chexground.model.roi_align import MLVLROIQueryModule, coordinate_to_encoding

SAFE_NORMALIZE_EPS = 1e-6


def build_1d_sincos_pos_embed(seq_len: int, dim: int, device, dtype):
    if dim % 2 != 0:
        raise ValueError(f"1D positional embedding expects even dim, got {dim}.")
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    dim_arr = torch.arange(0, dim, 2, device=device, dtype=dtype)
    div_term = torch.exp(dim_arr * (-math.log(10000.0) / dim))
    pos_emb = torch.zeros((seq_len, dim), device=device, dtype=dtype)
    pos_emb[:, 0::2] = torch.sin(pos[:, None] * div_term)
    pos_emb[:, 1::2] = torch.cos(pos[:, None] * div_term)
    return pos_emb


def build_2d_sincos_pos_embed(height: int, width: int, dim: int, device, dtype):
    if dim % 2 != 0:
        raise ValueError(f"2D positional embedding expects even dim, got {dim}.")
    y, x = torch.meshgrid(torch.linspace(0.0, 1.0, height, device=device, dtype=dtype), torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij")
    coord_grid = torch.stack([x, y], dim=-1).view(1, height * width, 2)
    pos_embed = coordinate_to_encoding(coord_grid, num_feats=dim // 2)
    return pos_embed


def masked_mean(tokens: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    tokens = tokens.masked_fill(~mask[..., None], 0.0)
    weights = mask.to(dtype=tokens.dtype)[..., None]
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (tokens * weights).sum(dim=1) / denom


def masked_multilabel_bce_per_image(logits: torch.Tensor, targets: torch.Tensor, label_mask: torch.BoolTensor) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    valid_mask = label_mask.to(dtype=torch.bool)
    valid_images = valid_mask.any(dim=1)
    if not valid_images.any():
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * valid_mask.to(dtype=loss.dtype)
    denom = valid_mask.sum(dim=1).clamp(min=1).to(dtype=loss.dtype)
    loss_per_image = loss.sum(dim=1) / denom
    return loss_per_image[valid_images].mean()


def mask_same_study_logits(pair_logits: torch.Tensor, same_study_mask: Optional[torch.BoolTensor]) -> torch.Tensor:
    if same_study_mask is None:
        return pair_logits
    mask = same_study_mask.to(device=pair_logits.device, dtype=torch.bool)
    if mask.numel() == 0 or not mask.any():
        return pair_logits
    return pair_logits.masked_fill(mask, torch.finfo(pair_logits.dtype).min)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim))

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        key_padding_mask = None
        effective_valid_mask = valid_mask
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
            effective_valid_mask = valid_mask.clone()
            no_valid_tokens = ~effective_valid_mask.any(dim=1)
            if no_valid_tokens.any():
                effective_valid_mask[no_valid_tokens, 0] = True
            x = x.masked_fill(~valid_mask[..., None], 0.0)
            key_padding_mask = ~effective_valid_mask
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask[..., None], 0.0)
        return x


class TemporalCarryBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, use_libra_prior_bias: bool = True):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.use_libra_prior_bias = bool(use_libra_prior_bias)
        self.num_causal_passes = 3
        self.temporal_causal_blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(dim), "attn": nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True),
                "post_norm": nn.LayerNorm(dim), "mlp": nn.Sequential(nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim))
            }) for _ in range(self.num_causal_passes)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim))

    def forward_causal_sequence(self, token_sequence: torch.Tensor, token_valid_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, temporal_length, tokens_per_step, dim = token_sequence.shape
        temporal_tokens = token_sequence.permute(0, 2, 1, 3).reshape(batch_size * tokens_per_step, temporal_length, dim)
        temporal_valid_mask = token_valid_mask.permute(0, 2, 1).reshape(batch_size * tokens_per_step, temporal_length)
        temporal_valid_mask = temporal_valid_mask.to(device=temporal_tokens.device, dtype=torch.bool)
        effective_temporal_valid_mask = temporal_valid_mask.clone()
        no_valid_tokens = ~effective_temporal_valid_mask.any(dim=1)
        if no_valid_tokens.any():
            effective_temporal_valid_mask[no_valid_tokens, 0] = True
        temporal_tokens = temporal_tokens.masked_fill(~temporal_valid_mask[..., None], 0.0)
        key_padding_mask = ~effective_temporal_valid_mask

        causal_mask = torch.ones(temporal_length, temporal_length, device=temporal_tokens.device, dtype=torch.bool).triu(1)

        for layer in self.temporal_causal_blocks:
            attn_input = layer["norm"](temporal_tokens)
            attn_output, _ = layer["attn"](query=attn_input, key=attn_input, value=attn_input, attn_mask=causal_mask,
                key_padding_mask=key_padding_mask, need_weights=False)
            temporal_tokens = temporal_tokens + attn_output
            temporal_tokens = temporal_tokens + layer["mlp"](layer["post_norm"](temporal_tokens))
            temporal_tokens = temporal_tokens.masked_fill(~temporal_valid_mask[..., None], 0.0)

        temporal_tokens = temporal_tokens.reshape(batch_size, tokens_per_step, temporal_length, dim).permute(0, 2, 1, 3)
        batch_size, temporal_length, tokens_per_step, dim = temporal_tokens.shape
        flat_tokens = temporal_tokens.reshape(batch_size * temporal_length, tokens_per_step, dim)
        flat_valid_mask = token_valid_mask.reshape(batch_size * temporal_length, tokens_per_step)
        # Refine individual tokens before downstream ROI pooling.
        refined_tokens = flat_tokens + self.mlp(self.final_norm(flat_tokens))
        flat_valid_mask = flat_valid_mask.to(device=refined_tokens.device, dtype=torch.bool)
        refined_tokens = refined_tokens.masked_fill(~flat_valid_mask[..., None], 0.0)
        return refined_tokens.reshape(batch_size, temporal_length, tokens_per_step, dim)


@dataclass
class GRPAOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict[str, torch.FloatTensor]] = None
    local_pair_logits: Optional[torch.FloatTensor] = None
    global_pair_logits: Optional[torch.FloatTensor] = None
    region_embeddings: Optional[torch.FloatTensor] = None
    global_embeddings: Optional[torch.FloatTensor] = None
    roi_last_hidden_state: Optional[torch.FloatTensor] = None
    global_last_hidden_state: Optional[torch.FloatTensor] = None
    text_embeddings: Optional[torch.FloatTensor] = None
    class_logits: Optional[torch.FloatTensor] = None
    temporal_class_logits: Optional[torch.FloatTensor] = None
    pred_boxes: Optional[torch.FloatTensor] = None
    roi_valid_mask: Optional[torch.BoolTensor] = None
    temporal_roi_last_hidden_state: Optional[torch.FloatTensor] = None
    temporal_global_last_hidden_state: Optional[torch.FloatTensor] = None


class GRPAConfig(PretrainedConfig):
    model_type = "grpa"

    def __init__(
        self, ddetr_checkpoint: Optional[str] = None, ddetr_cfg: Optional[Union[dict, CustomDDETRConfig]] = None,
        text_encoder_name: Optional[str] = "microsoft/BiomedVLP-BioViL-T", text_encoder_cfg: Optional[Union[dict, PretrainedConfig]] = None,
        num_classes: int = 14, roi_output_size: int = 14, roi_sampling_ratio: int = 2, roi_num_heads: int = 8, global_num_heads: int = 8,
        roi_mlp_ratio: float = 4.0, global_mlp_ratio: float = 4.0, jitter_translate: float = 0.05, jitter_scale: float = 0.05, jitter_expand_max: float = 0.3,
        text_max_length: int = 96, cls_weight: float = 0.25, local_gloria_weight: float = 1.0, local_aux_weight: float = 1.0,
        global_contrastive_weight: float = 1.0, local_aux_type: str = "cosine", attn_kl_soft_alpha: float = 0.8, temp1: float = 4.0, temp2: float = 5.0,
        temp3: float = 10.0, temporal_num_heads: Optional[int] = None, temporal_mlp_ratio: Optional[float] = None, temporal_use_libra_prior_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if local_aux_type not in {"cosine", "attn_kl"}:
            raise ValueError(f"local_aux_type must be one of {{'cosine', 'attn_kl'}}, got {local_aux_type}.")
        if not 0.0 <= attn_kl_soft_alpha <= 1.0:
            raise ValueError(f"attn_kl_soft_alpha must be in [0, 1], got {attn_kl_soft_alpha}.")
        if jitter_expand_max < 0.0:
            raise ValueError(f"jitter_expand_max must be non-negative, got {jitter_expand_max}.")
        if temporal_num_heads is not None and temporal_num_heads <= 0:
            raise ValueError(f"temporal_num_heads must be positive when provided, got {temporal_num_heads}.")
        if temporal_mlp_ratio is not None and temporal_mlp_ratio <= 0.0:
            raise ValueError(f"temporal_mlp_ratio must be positive when provided, got {temporal_mlp_ratio}.")

        self.ddetr_checkpoint = ddetr_checkpoint
        if isinstance(ddetr_cfg, CustomDDETRConfig):
            self.ddetr_cfg = ddetr_cfg
        elif isinstance(ddetr_cfg, dict):
            payload = dict(ddetr_cfg)
            payload.pop("model_type", None)
            self.ddetr_cfg = CustomDDETRConfig(**payload)
        elif ddetr_cfg is not None:
            raise TypeError(f"Unsupported ddetr_cfg type: {type(ddetr_cfg)!r}")
        else:
            self.ddetr_cfg = CustomDDETRConfig.from_pretrained(ddetr_checkpoint) if ddetr_checkpoint else None

        self.text_encoder_name = text_encoder_name
        if isinstance(text_encoder_cfg, PretrainedConfig):
            self.text_encoder_cfg = text_encoder_cfg
        elif isinstance(text_encoder_cfg, dict):
            payload = dict(text_encoder_cfg)
            model_type = payload.pop("model_type", None)
            if model_type:
                try:
                    text_encoder_cfg = AutoConfig.for_model(model_type, **payload)
                except Exception:
                    pass
            if isinstance(text_encoder_cfg, dict):
                if not text_encoder_name:
                    raise ValueError("Could not reconstruct text_encoder_cfg from config dict without text_encoder_name fallback.")
                text_encoder_cfg = AutoConfig.from_pretrained(text_encoder_name, trust_remote_code=True)
            self.text_encoder_cfg = text_encoder_cfg
        elif text_encoder_cfg is not None:
            raise TypeError(f"Unsupported text_encoder_cfg type: {type(text_encoder_cfg)!r}")
        else:
            self.text_encoder_cfg = AutoConfig.from_pretrained(text_encoder_name, trust_remote_code=True) if text_encoder_name else None

        self.num_classes = num_classes
        self.roi_output_size = roi_output_size
        self.roi_sampling_ratio = roi_sampling_ratio
        self.roi_num_heads = roi_num_heads
        self.global_num_heads = global_num_heads
        self.roi_mlp_ratio = roi_mlp_ratio
        self.global_mlp_ratio = global_mlp_ratio
        self.jitter_translate = jitter_translate
        self.jitter_scale = jitter_scale
        self.jitter_expand_max = jitter_expand_max
        self.text_max_length = text_max_length
        self.cls_weight = cls_weight
        self.local_gloria_weight = local_gloria_weight
        self.local_aux_weight = local_aux_weight
        self.global_contrastive_weight = global_contrastive_weight
        self.local_aux_type = local_aux_type
        self.attn_kl_soft_alpha = attn_kl_soft_alpha
        self.temp1 = temp1
        self.temp2 = temp2
        self.temp3 = temp3
        self.temporal_num_heads = temporal_num_heads
        self.temporal_mlp_ratio = temporal_mlp_ratio
        self.temporal_use_libra_prior_bias = temporal_use_libra_prior_bias

    def to_json_string(self, use_diff: bool = True) -> str:
        config_dict = self.to_diff_dict() if use_diff else self.to_dict()
        config_dict["ddetr_cfg"] = (None if self.ddetr_cfg is None else json.loads(self.ddetr_cfg.to_json_string(use_diff=use_diff)))
        config_dict["text_encoder_cfg"] = (None if self.text_encoder_cfg is None else json.loads(self.text_encoder_cfg.to_json_string(use_diff=use_diff)))
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"


class GRPA(PreTrainedModel):
    """ Equivalent to Temporal Region-Phrase Alignment (TRPA) model in the paper"""
    config_class = GRPAConfig
    MIN_VALID_ROI_AREA = 0.0001092283

    def __init__(self, config: GRPAConfig):
        super().__init__(config)

        if config.ddetr_cfg is None:
            raise ValueError("GRPAConfig must provide ddetr_cfg or ddetr_checkpoint.")
        if config.text_encoder_cfg is None:
            raise ValueError("GRPAConfig must provide text_encoder_cfg or text_encoder_name.")

        self.detector = CustomDDETRModel(config.ddetr_cfg)
        self.detector.requires_grad_(False)
        self.detector.eval()

        self.text_encoder = AutoModel.from_config(config.text_encoder_cfg, trust_remote_code=True)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        ddetr_cfg = self.detector.config.ddetr_cfg
        self.num_slots = int(getattr(ddetr_cfg, "anatomy_num_queries", ddetr_cfg.num_queries))
        self.num_feature_levels = int(ddetr_cfg.num_feature_levels)
        self.visual_dim = 512
        self.text_embed_dim = int(getattr(self.text_encoder.config, "projection_size", 128))
        self.backbone_hidden_dim = int(self.detector.config.vis_encoder_cfg.hidden_size)

        self.global_img_txt_bridge = nn.Sequential(nn.Linear(self.backbone_hidden_dim * 4, self.visual_dim), nn.GELU(),
            nn.Linear(self.visual_dim, self.visual_dim))
        self.roi_backbone_downsample = nn.Sequential(nn.Linear(self.backbone_hidden_dim, self.visual_dim), nn.LayerNorm(self.visual_dim), nn.GELU())
        self.roi_fuse_proj = nn.Sequential(nn.Conv2d(self.visual_dim, self.visual_dim, kernel_size=3, padding=1), LayerNorm(self.visual_dim), nn.GELU())
        self.roi_align = MLVLROIQueryModule(embed_dims=self.visual_dim, out_dims=self.visual_dim, num_levels=self.num_feature_levels, return_spatial=True,
            roi_output_size=config.roi_output_size, roi_sampling_ratio=config.roi_sampling_ratio)
        self.roi_blocks = nn.ModuleList([SelfAttentionBlock(self.visual_dim, config.roi_num_heads, config.roi_mlp_ratio) for _ in range(2)])

        self.inter_roi_token_attn = SelfAttentionBlock(self.visual_dim, config.roi_num_heads, config.roi_mlp_ratio)
        self.roi_box_pos_proj = nn.Sequential(nn.Linear(4, 256), nn.ReLU(inplace=True), nn.LayerNorm(256), nn.Linear(256, self.visual_dim),
            nn.ReLU(inplace=True), nn.LayerNorm(self.visual_dim))
        self.roi_projection = nn.Sequential(nn.LayerNorm(self.visual_dim), nn.Linear(self.visual_dim, self.text_embed_dim))
        self.local_phrase_projection = nn.Sequential(nn.LayerNorm(self.text_embed_dim), nn.Linear(self.text_embed_dim, self.text_embed_dim), nn.GELU(),
            nn.Linear(self.text_embed_dim, self.text_embed_dim))

        self.global_self_pre = SelfAttentionBlock(self.visual_dim, config.global_num_heads, config.global_mlp_ratio)
        self.global_self_post = SelfAttentionBlock(self.visual_dim, config.global_num_heads, config.global_mlp_ratio)
        self.global_projection = nn.Sequential(nn.LayerNorm(self.visual_dim), nn.Linear(self.visual_dim, self.text_embed_dim))
        temporal_roi_num_heads = config.temporal_num_heads or config.roi_num_heads
        temporal_roi_mlp_ratio = config.temporal_mlp_ratio or config.roi_mlp_ratio
        self.temporal_roi_carry = TemporalCarryBlock(self.visual_dim, temporal_roi_num_heads, temporal_roi_mlp_ratio,
            use_libra_prior_bias=config.temporal_use_libra_prior_bias)
        if config.cls_weight > 0.0:
            self.mil_attn_hidden = nn.Linear(self.text_embed_dim, self.text_embed_dim)
            self.mil_attn_score = nn.Linear(self.text_embed_dim, config.num_classes)
            self.mil_classifier_weight = nn.Parameter(torch.empty(config.num_classes, self.text_embed_dim))
            self.mil_classifier_bias = nn.Parameter(torch.zeros(config.num_classes))

            nn.init.normal_(self.mil_classifier_weight, std=0.02)
        self.local_matcher = GLoRIAROILocalMatcher()
        self.latest_loss_dict = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.detector.eval()
        self.text_encoder.eval()
        return self

    def _build_backbone_feature_maps(self, hidden_states):
        if self.num_feature_levels == 1:
            select_hidden_state = torch.stack(hidden_states[-4:])
            image_features = torch.mean(select_hidden_state, dim=0)[:, 1:]
            batch_size, sequence_length, hidden_dim = image_features.shape
            height = width = int(math.sqrt(sequence_length))
            return [image_features.reshape(batch_size, height, width, hidden_dim).permute(0, 3, 1, 2).contiguous()]

        if self.num_feature_levels == 3:
            backbone_feature_maps = []
            for hidden_state in hidden_states[-3:]:
                hidden_state = hidden_state[:, 1:]
                batch_size, sequence_length, hidden_dim = hidden_state.shape
                height = width = int(math.sqrt(sequence_length))
                backbone_feature_maps.append(hidden_state.reshape(batch_size, height, width, hidden_dim).permute(0, 3, 1, 2).contiguous())
            return backbone_feature_maps

        raise ValueError(f"Unsupported num_feature_levels={self.num_feature_levels}.")

    def _downsample_backbone_feature_maps_for_roi(self, backbone_feature_maps: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        selected_feature_maps = tuple(backbone_feature_maps[-self.num_feature_levels:])
        # Project all feature levels together in their existing batch order.
        stacked_feature_maps = torch.stack(selected_feature_maps, dim=1)
        batch_size, num_levels, channels, height, width = stacked_feature_maps.shape
        stacked_token_maps = stacked_feature_maps.permute(0, 1, 3, 4, 2).reshape(batch_size * num_levels, height * width, channels)
        stacked_token_maps = self.roi_backbone_downsample(stacked_token_maps)
        stacked_downsampled_feature_maps = stacked_token_maps.reshape(batch_size, num_levels, height, width, self.visual_dim).permute(0, 1, 4, 2, 3)
        return tuple(feature_map.contiguous() for feature_map in stacked_downsampled_feature_maps.unbind(dim=1))

    def _run_frozen_detector(self, images: torch.Tensor, pixel_mask: torch.BoolTensor, backbone_outputs=None):
        if backbone_outputs is None:
            with torch.no_grad():
                backbone_outputs = self.detector.vis_encoder(images, output_hidden_states=True)
        if isinstance(backbone_outputs, dict):
            backbone_outputs = backbone_outputs.get("image_forward_outs", backbone_outputs)
        hidden_states = tuple(hidden_state.detach().contiguous() for hidden_state in backbone_outputs.hidden_states)
        with torch.no_grad():
            backbone_feature_maps = self._build_backbone_feature_maps(hidden_states)
            roi_backbone_feature_maps = self._downsample_backbone_feature_maps_for_roi(tuple(backbone_feature_maps))
            select_hidden_state_layer = getattr(self.detector.config, "vis_output_layer", -1)
            global_backbone_tokens = hidden_states[select_hidden_state_layer][:, 1:]
            srcs = [input_proj(feature_map) for input_proj, feature_map in zip(self.detector.input_proj, backbone_feature_maps)]
            masks = [F.interpolate(pixel_mask[:, None].float(), size=src.shape[-2:], mode="nearest")[:, 0] > 0.5 for src in srcs]
            ddetr_outputs = self.detector.ddetr_transformer(sources=srcs, masks=masks, labels=None, output_attentions=False, output_hidden_states=False,
                return_dict=True)

        return {"backbone_feature_maps": tuple(backbone_feature_maps), "roi_backbone_feature_maps": roi_backbone_feature_maps,
            "global_backbone_tokens": global_backbone_tokens, "srcs": srcs, "masks": masks, "ddetr_outputs": ddetr_outputs}

    def _apply_box_jitter(self, roi_boxes: torch.Tensor) -> torch.Tensor:
        if (not self.training or (self.config.jitter_translate <= 0 and self.config.jitter_scale <= 0 and self.config.jitter_expand_max <= 0)):
            return roi_boxes

        cx, cy, bw, bh = roi_boxes.unbind(dim=-1)
        roi_area = (bw * bh).clamp(min=0.0, max=1.0)
        small_roi_weight = 1.0 - torch.sqrt(roi_area)
        if self.config.jitter_translate > 0:
            translate_x = (torch.rand_like(cx) * 2.0 - 1.0) * self.config.jitter_translate * bw * small_roi_weight
            translate_y = (torch.rand_like(cy) * 2.0 - 1.0) * self.config.jitter_translate * bh * small_roi_weight
            cx = (cx + translate_x).clamp(min=0.0, max=1.0)
            cy = (cy + translate_y).clamp(min=0.0, max=1.0)
        expand_ratio = torch.rand_like(bw) * self.config.jitter_expand_max * small_roi_weight
        scale_w = (1.0 + expand_ratio) * (1.0 + (torch.rand_like(bw) * 2.0 - 1.0) * self.config.jitter_scale)
        scale_h = (1.0 + expand_ratio) * (1.0 + (torch.rand_like(bh) * 2.0 - 1.0) * self.config.jitter_scale)

        bw = (bw * scale_w.clamp(min=1e-3)).clamp(min=1e-4, max=1.0)
        bh = (bh * scale_h.clamp(min=1e-3)).clamp(min=1e-4, max=1.0)
        return torch.stack([cx, cy, bw, bh], dim=-1)

    def _sanitize_roi_boxes(self, roi_boxes: torch.Tensor, pixel_mask: torch.BoolTensor) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """Clip boxes to the valid image rectangle and filter by minimum area."""
        batch_size, num_regions, _ = roi_boxes.shape
        image_height, image_width = pixel_mask.shape[-2:]
        roi_boxes = roi_boxes.detach().clone()
        pixel_mask = pixel_mask.to(device=roi_boxes.device, dtype=torch.bool)

        valid_rows = pixel_mask.any(dim=2)
        valid_cols = pixel_mask.any(dim=1)
        valid_top = roi_boxes.new_zeros((batch_size,))
        valid_bottom = roi_boxes.new_ones((batch_size,))
        valid_left = roi_boxes.new_zeros((batch_size,))
        valid_right = roi_boxes.new_ones((batch_size,))

        for batch_idx in range(batch_size):
            row_indices = torch.nonzero(valid_rows[batch_idx], as_tuple=False).flatten()
            col_indices = torch.nonzero(valid_cols[batch_idx], as_tuple=False).flatten()
            if row_indices.numel() == 0 or col_indices.numel() == 0:
                continue
            valid_top[batch_idx] = row_indices[0].to(dtype=roi_boxes.dtype) / image_height
            valid_bottom[batch_idx] = (row_indices[-1] + 1).to(dtype=roi_boxes.dtype) / image_height
            valid_left[batch_idx] = col_indices[0].to(dtype=roi_boxes.dtype) / image_width
            valid_right[batch_idx] = (col_indices[-1] + 1).to(dtype=roi_boxes.dtype) / image_width

        cx, cy, bw, bh = roi_boxes.unbind(dim=-1)
        valid_top = valid_top[:, None]
        valid_bottom = valid_bottom[:, None]
        valid_left = valid_left[:, None]
        valid_right = valid_right[:, None]
        min_half_w = roi_boxes.new_full((batch_size, 1), 0.5 / image_width)
        min_half_h = roi_boxes.new_full((batch_size, 1), 0.5 / image_height)

        center_min_x = valid_left + min_half_w
        center_max_x = valid_right - min_half_w
        center_min_y = valid_top + min_half_h
        center_max_y = valid_bottom - min_half_h
        fallback_cx = (valid_left + valid_right) / 2.0
        fallback_cy = (valid_top + valid_bottom) / 2.0

        cx = torch.where(center_min_x <= center_max_x, cx.clamp(min=center_min_x, max=center_max_x), fallback_cx)
        cy = torch.where(center_min_y <= center_max_y, cy.clamp(min=center_min_y, max=center_max_y), fallback_cy)

        max_half_w = torch.minimum(cx - valid_left, valid_right - cx).clamp(min=min_half_w)
        max_half_h = torch.minimum(cy - valid_top, valid_bottom - cy).clamp(min=min_half_h)
        bw = torch.minimum(bw.clamp(min=2.0 * min_half_w), 2.0 * max_half_w)
        bh = torch.minimum(bh.clamp(min=2.0 * min_half_h), 2.0 * max_half_h)
        roi_boxes = torch.stack([cx, cy, bw, bh], dim=-1)

        roi_area = roi_boxes[..., 2] * roi_boxes[..., 3]
        roi_valid_mask = roi_area >= self.MIN_VALID_ROI_AREA
        return roi_boxes, roi_valid_mask

    def _encode_roi_tokens(
        self, backbone_feature_maps: Tuple[torch.Tensor, ...], roi_boxes: torch.Tensor, roi_valid_mask: torch.BoolTensor, image_shapes: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        roi_box_list = [roi_boxes[batch_idx] for batch_idx in range(roi_boxes.shape[0])]
        spatial_features = self.roi_align(list(backbone_feature_maps), roi_box_list, image_shapes=image_shapes)
        spatial_features = torch.stack(spatial_features, dim=0)
        fused_roi_features = spatial_features.mean(dim=2)

        B, R, C, H, W = fused_roi_features.shape
        fused_roi_features = fused_roi_features.reshape(B * R, C, H, W)
        fused_roi_features = self.roi_fuse_proj(fused_roi_features)
        fused_channels = fused_roi_features.shape[1]
        fused_roi_features = fused_roi_features.reshape(B, R, fused_channels, H, W)

        batch_size, num_regions, channels, height, width = fused_roi_features.shape
        spatial_pos = build_2d_sincos_pos_embed(height, width, channels, fused_roi_features.device, fused_roi_features.dtype)
        flat_boxes = roi_boxes.reshape(batch_size * num_regions, 4)
        bbox_pos = self.roi_box_pos_proj(flat_boxes)

        roi_tokens = fused_roi_features.reshape(batch_size * num_regions, channels, height * width).transpose(1, 2)
        roi_tokens = roi_tokens + spatial_pos.to(dtype=roi_tokens.dtype)
        roi_tokens = roi_tokens + bbox_pos.reshape(batch_size * num_regions, 1, channels)

        region_token_valid_mask = roi_valid_mask.reshape(batch_size * num_regions, 1).expand(-1, roi_tokens.shape[1])
        for i, block in enumerate(self.roi_blocks):
            roi_tokens = block(roi_tokens, valid_mask=region_token_valid_mask)

            if i == 0:
                num_spatial_tokens = roi_tokens.shape[1]
                token_dim = roi_tokens.shape[2]

                mixed_roi_tokens = roi_tokens.reshape(batch_size, num_regions, num_spatial_tokens, token_dim)
                mixed_roi_tokens = mixed_roi_tokens.reshape(batch_size, num_regions * num_spatial_tokens, token_dim)

                inter_roi_valid_mask = region_token_valid_mask.reshape(batch_size, num_regions, num_spatial_tokens)
                inter_roi_valid_mask = inter_roi_valid_mask.reshape(batch_size, num_regions * num_spatial_tokens)

                mixed_roi_tokens = self.inter_roi_token_attn(mixed_roi_tokens, valid_mask=inter_roi_valid_mask)

                roi_tokens = mixed_roi_tokens.reshape(batch_size, num_regions, num_spatial_tokens, token_dim)
                roi_tokens = roi_tokens.reshape(batch_size * num_regions, num_spatial_tokens, token_dim)

        roi_last_hidden_state = roi_tokens.reshape(batch_size, num_regions, roi_tokens.shape[1], roi_tokens.shape[2])
        roi_last_hidden_state = roi_last_hidden_state.masked_fill(~roi_valid_mask[:, :, None, None], 0.0)
        return roi_last_hidden_state, None

    def _encode_text_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_ids is None or input_ids.shape[0] == 0:
            device = attention_mask.device if attention_mask is not None else self.device
            return torch.zeros((0, self.text_embed_dim), dtype=torch.float32, device=device)

        encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoded["token_type_ids"] = token_type_ids
        with torch.no_grad():
            if hasattr(self.text_encoder, "get_projected_text_embeddings"):
                text_embeddings = self.text_encoder.get_projected_text_embeddings(encoded["input_ids"], encoded["attention_mask"])
            else:
                outputs = self.text_encoder(**encoded, output_cls_projected_embedding=True, return_dict=True)
                text_embeddings = outputs.cls_projected_embedding
        return F.normalize(text_embeddings.float(), dim=-1, eps=SAFE_NORMALIZE_EPS)

    def _normalize_temporal_inputs(self, images: torch.Tensor, pixel_mask: torch.BoolTensor) -> Tuple[torch.Tensor, torch.BoolTensor]:
        if images.dim() == 4:
            images = images[:, None]
        elif images.dim() != 5:
            raise ValueError(f"Expected images to have 4 or 5 dims, got shape={tuple(images.shape)}.")

        if pixel_mask.dim() == 3:
            pixel_mask = pixel_mask[:, None]
        elif pixel_mask.dim() != 4:
            raise ValueError(f"Expected pixel_mask to have 3 or 4 dims, got shape={tuple(pixel_mask.shape)}.")

        if images.shape[:2] != pixel_mask.shape[:2]:
            raise ValueError(
                "images and pixel_mask must match on batch/time axes, "
                f"got images.shape[:2]={tuple(images.shape[:2])} pixel_mask.shape[:2]={tuple(pixel_mask.shape[:2])}."
            )
        return images, pixel_mask.to(dtype=torch.bool, device=images.device)

    def _project_region_embeddings(self, roi_tokens: torch.Tensor, roi_valid_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, num_regions, num_roi_tokens, _ = roi_tokens.shape
        if num_roi_tokens == 1:
            pooled_roi_tokens = roi_tokens[:, :, 0, :]
        else:
            region_token_valid_mask = roi_valid_mask[:, :, None].expand(-1, -1, num_roi_tokens)
            pooled_roi_tokens = masked_mean(roi_tokens.reshape(batch_size * num_regions, num_roi_tokens, roi_tokens.shape[-1]),
                region_token_valid_mask.reshape(batch_size * num_regions, num_roi_tokens)).reshape(batch_size, num_regions, -1)
        region_embeddings = self.roi_projection(pooled_roi_tokens)
        region_embeddings = F.normalize(region_embeddings, dim=-1, eps=SAFE_NORMALIZE_EPS)
        return region_embeddings.masked_fill(~roi_valid_mask[..., None], 0.0)

    def _apply_temporal_carry(self, token_sequence: torch.Tensor, token_valid_mask: torch.BoolTensor, carry_block: TemporalCarryBlock) -> torch.Tensor:
        token_valid_mask = token_valid_mask.to(device=token_sequence.device, dtype=torch.bool)
        temporal_length = token_sequence.shape[1]
        dim = token_sequence.shape[-1]

        temporal_pos_embed = build_1d_sincos_pos_embed(temporal_length, dim, token_sequence.device, token_sequence.dtype)
        token_sequence = token_sequence + temporal_pos_embed.view(1, temporal_length, 1, dim)
        return carry_block.forward_causal_sequence(token_sequence, token_valid_mask)

    def _compute_mil_logits(self, region_embeddings: torch.Tensor, roi_valid_mask: torch.BoolTensor) -> torch.Tensor:
        effective_roi_valid_mask = roi_valid_mask.clone()
        no_valid_roi = ~effective_roi_valid_mask.any(dim=1)
        if no_valid_roi.any():
            effective_roi_valid_mask[no_valid_roi, 0] = True

        attn_hidden = torch.tanh(self.mil_attn_hidden(region_embeddings))
        attn_scores = self.mil_attn_score(attn_hidden)
        invalid_mask = ~effective_roi_valid_mask[:, :, None]
        attn_scores = attn_scores.masked_fill(invalid_mask, torch.finfo(attn_scores.dtype).min)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.einsum("bkc,bkd->bcd", attn_weights, region_embeddings)
        logits = (pooled * self.mil_classifier_weight[None]).sum(dim=-1) + self.mil_classifier_bias[None]
        return logits

    def extract_roi_features(
        self, images: torch.Tensor, pixel_mask: torch.BoolTensor, temporal_lengths: Optional[torch.LongTensor] = None, backbone_outputs=None,
    ) -> Dict[str, torch.Tensor]:
        images, pixel_mask = self._normalize_temporal_inputs(images, pixel_mask)
        batch_size, temporal_length = images.shape[:2]
        if temporal_lengths is None:
            temporal_lengths = torch.full((batch_size,), temporal_length, device=images.device, dtype=torch.long)
        else:
            temporal_lengths = temporal_lengths.to(device=images.device, dtype=torch.long)
        temporal_lengths = temporal_lengths.to(device=images.device, dtype=torch.long)
        time_index = torch.arange(temporal_length, device=images.device, dtype=torch.long).view(1, temporal_length)
        first_valid_index = temporal_length - temporal_lengths.view(temporal_lengths.shape[0], 1)
        valid_frame_mask = time_index >= first_valid_index
        flat_valid_mask = valid_frame_mask.reshape(-1)
        flat_images_all = images.reshape(batch_size * temporal_length, *images.shape[2:])
        flat_pixel_mask_all = pixel_mask.reshape(batch_size * temporal_length, *pixel_mask.shape[2:])
        flat_images = flat_images_all[flat_valid_mask]
        flat_pixel_mask = flat_pixel_mask_all[flat_valid_mask]

        frozen_outputs = self._run_frozen_detector(images=flat_images, pixel_mask=flat_pixel_mask, backbone_outputs=backbone_outputs)
        ddetr_outputs = frozen_outputs["ddetr_outputs"]
        anatomy_boxes = ddetr_outputs.pred_boxes["anatomy"].detach()
        roi_boxes = anatomy_boxes
        roi_boxes, roi_valid_mask = self._sanitize_roi_boxes(roi_boxes, flat_pixel_mask)

        roi_backbone_feature_maps = frozen_outputs["roi_backbone_feature_maps"]
        frame_roi_last_hidden_state, _ = self._encode_roi_tokens(backbone_feature_maps=roi_backbone_feature_maps, roi_boxes=roi_boxes,
            roi_valid_mask=roi_valid_mask, image_shapes=flat_images.shape[-2:])
        num_roi_tokens = frame_roi_last_hidden_state.shape[2]
        dense_roi_boxes = roi_boxes.new_zeros((batch_size * temporal_length, self.num_slots, 4))
        dense_roi_valid_mask = torch.zeros((batch_size * temporal_length, self.num_slots), dtype=torch.bool, device=roi_valid_mask.device)
        dense_frame_roi_last_hidden_state = frame_roi_last_hidden_state.new_zeros(
            (batch_size * temporal_length, self.num_slots, num_roi_tokens, self.visual_dim))
        flat_valid_mask_device = flat_valid_mask.to(device=roi_boxes.device)
        dense_roi_boxes[flat_valid_mask_device] = roi_boxes
        dense_roi_valid_mask[flat_valid_mask_device] = roi_valid_mask
        dense_frame_roi_last_hidden_state[flat_valid_mask_device] = frame_roi_last_hidden_state

        temporal_roi_boxes = dense_roi_boxes.reshape(batch_size, temporal_length, self.num_slots, 4)
        temporal_roi_valid_mask = dense_roi_valid_mask.reshape(batch_size, temporal_length, self.num_slots)
        temporal_roi_tokens = dense_frame_roi_last_hidden_state.reshape(batch_size, temporal_length, self.num_slots, num_roi_tokens, self.visual_dim)
        temporal_roi_token_valid_mask = temporal_roi_valid_mask[:, :, :, None].expand(-1, -1, -1, num_roi_tokens)
        roi_scan_tokens = temporal_roi_tokens.permute(0, 2, 1, 3, 4).reshape(batch_size * self.num_slots, temporal_length, num_roi_tokens, self.visual_dim)
        roi_scan_valid_mask = temporal_roi_token_valid_mask.permute(0, 2, 1, 3).reshape(batch_size * self.num_slots, temporal_length, num_roi_tokens)
        temporal_roi_last_hidden_state = self._apply_temporal_carry(token_sequence=roi_scan_tokens, token_valid_mask=roi_scan_valid_mask,
            carry_block=self.temporal_roi_carry)
        temporal_roi_last_hidden_state = temporal_roi_last_hidden_state.reshape(batch_size, self.num_slots, temporal_length,
            num_roi_tokens, self.visual_dim).permute(0, 2, 1, 3, 4)
        current_roi_last_hidden_state = temporal_roi_last_hidden_state[:, -1]
        current_roi_valid_mask = temporal_roi_valid_mask[:, -1]
        current_roi_token_valid_mask = temporal_roi_token_valid_mask[:, -1]
        current_roi_boxes = temporal_roi_boxes[:, -1]

        return {"images": images, "pixel_mask": pixel_mask, "temporal_lengths": temporal_lengths, "valid_frame_mask": valid_frame_mask,
            "temporal_roi_last_hidden_state": temporal_roi_last_hidden_state, "temporal_roi_valid_mask": temporal_roi_valid_mask,
            "temporal_roi_token_valid_mask": temporal_roi_token_valid_mask, "temporal_roi_boxes": temporal_roi_boxes,
            "roi_last_hidden_state": current_roi_last_hidden_state, "roi_valid_mask": current_roi_valid_mask,
            "roi_token_valid_mask": current_roi_token_valid_mask, "pred_boxes": current_roi_boxes}

    def forward(
        self, images: torch.Tensor, pixel_mask: torch.BoolTensor, image_labels: Optional[torch.FloatTensor] = None,
        image_label_mask: Optional[torch.BoolTensor] = None, temporal_image_labels: Optional[torch.FloatTensor] = None,
        temporal_image_label_mask: Optional[torch.BoolTensor] = None, phrase_input_ids: Optional[torch.LongTensor] = None,
        phrase_attention_mask: Optional[torch.LongTensor] = None, phrase_token_type_ids: Optional[torch.LongTensor] = None,
        report_input_ids: Optional[torch.LongTensor] = None, report_attention_mask: Optional[torch.LongTensor] = None,
        report_token_type_ids: Optional[torch.LongTensor] = None, report_valid_mask: Optional[torch.BoolTensor] = None,
        phrase_batch_index: Optional[torch.LongTensor] = None, phrase_positive_roi_mask: Optional[torch.BoolTensor] = None,
        same_study_mask: Optional[torch.BoolTensor] = None, return_dict: Optional[bool] = None,
    ) -> Union[Tuple, GRPAOutput]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        images, pixel_mask = self._normalize_temporal_inputs(images, pixel_mask)
        batch_size, temporal_length = images.shape[:2]
        flat_images = images.reshape(batch_size * temporal_length, *images.shape[2:])
        flat_pixel_mask = pixel_mask.reshape(batch_size * temporal_length, *pixel_mask.shape[2:])
        if image_labels is not None:
            image_labels = image_labels.to(device=flat_images.device, dtype=torch.float32)
        if image_label_mask is not None:
            image_label_mask = image_label_mask.to(device=flat_images.device, dtype=torch.bool)
        if temporal_image_labels is not None:
            temporal_image_labels = temporal_image_labels.to(device=flat_images.device, dtype=torch.float32)
        if temporal_image_label_mask is not None:
            temporal_image_label_mask = temporal_image_label_mask.to(device=flat_images.device, dtype=torch.bool)
        if same_study_mask is not None:
            same_study_mask = same_study_mask.to(device=flat_images.device, dtype=torch.bool)
        if phrase_input_ids is None:
            phrase_input_ids = torch.zeros((0, 1), dtype=torch.long, device=flat_images.device)
        else:
            phrase_input_ids = phrase_input_ids.to(device=flat_images.device, dtype=torch.long)
        if phrase_attention_mask is None:
            phrase_attention_mask = torch.zeros_like(phrase_input_ids)
        else:
            phrase_attention_mask = phrase_attention_mask.to(device=flat_images.device, dtype=torch.long)
        if phrase_token_type_ids is not None:
            phrase_token_type_ids = phrase_token_type_ids.to(device=flat_images.device, dtype=torch.long)
        if phrase_batch_index is None:
            phrase_batch_index = torch.zeros((0,), dtype=torch.long, device=flat_images.device)
        else:
            phrase_batch_index = phrase_batch_index.to(device=flat_images.device, dtype=torch.long)
        if phrase_positive_roi_mask is None:
            phrase_positive_roi_mask = torch.zeros((phrase_input_ids.shape[0], self.num_slots), dtype=torch.bool, device=flat_images.device)
        else:
            phrase_positive_roi_mask = phrase_positive_roi_mask.to(device=flat_images.device, dtype=torch.bool)

        frozen_outputs = self._run_frozen_detector(images=flat_images, pixel_mask=flat_pixel_mask)
        ddetr_outputs = frozen_outputs["ddetr_outputs"]
        anatomy_boxes = ddetr_outputs.pred_boxes["anatomy"].detach()
        roi_boxes = self._apply_box_jitter(anatomy_boxes)
        roi_boxes, roi_valid_mask = self._sanitize_roi_boxes(roi_boxes, flat_pixel_mask)
        temporal_roi_boxes = roi_boxes.reshape(batch_size, temporal_length, self.num_slots, 4)
        temporal_roi_valid_mask = roi_valid_mask.reshape(batch_size, temporal_length, self.num_slots)

        roi_backbone_feature_maps = frozen_outputs["roi_backbone_feature_maps"]

        frame_roi_last_hidden_state, _ = self._encode_roi_tokens(backbone_feature_maps=roi_backbone_feature_maps, roi_boxes=roi_boxes,
            roi_valid_mask=roi_valid_mask, image_shapes=flat_images.shape[-2:])
        num_roi_tokens = frame_roi_last_hidden_state.shape[2]
        temporal_roi_tokens = frame_roi_last_hidden_state.reshape(batch_size, temporal_length, self.num_slots, num_roi_tokens, self.visual_dim)
        temporal_roi_token_valid_mask = temporal_roi_valid_mask[:, :, :, None].expand(-1, -1, -1, num_roi_tokens)
        roi_scan_tokens = temporal_roi_tokens.permute(0, 2, 1, 3, 4).reshape(batch_size * self.num_slots, temporal_length, num_roi_tokens, self.visual_dim)
        roi_scan_valid_mask = temporal_roi_token_valid_mask.permute(0, 2, 1, 3).reshape(batch_size * self.num_slots, temporal_length, num_roi_tokens)
        temporal_roi_last_hidden_state = self._apply_temporal_carry(token_sequence=roi_scan_tokens, token_valid_mask=roi_scan_valid_mask,
            carry_block=self.temporal_roi_carry)
        temporal_roi_last_hidden_state = temporal_roi_last_hidden_state.reshape(batch_size, self.num_slots, temporal_length,
            num_roi_tokens, self.visual_dim).permute(0, 2, 1, 3, 4)
        current_roi_last_hidden_state = temporal_roi_last_hidden_state[:, -1]
        current_roi_valid_mask = temporal_roi_valid_mask[:, -1]
        current_roi_boxes = temporal_roi_boxes[:, -1]
        region_embeddings = self._project_region_embeddings(current_roi_last_hidden_state, current_roi_valid_mask)
        phrase_embeddings = self._encode_text_inputs(input_ids=phrase_input_ids, attention_mask=phrase_attention_mask, token_type_ids=phrase_token_type_ids)
        projected_phrase_embeddings = phrase_embeddings
        if phrase_embeddings.numel() != 0:
            projected_phrase_embeddings = self.local_phrase_projection(phrase_embeddings)
            projected_phrase_embeddings = F.normalize(phrase_embeddings + projected_phrase_embeddings, dim=-1, eps=SAFE_NORMALIZE_EPS)

        local_match_outputs = self.local_matcher(region_embeddings=region_embeddings, roi_valid_mask=current_roi_valid_mask,
            phrase_attention_embeddings=projected_phrase_embeddings, phrase_score_embeddings=phrase_embeddings, phrase_batch_index=phrase_batch_index,
            temp1=self.config.temp1, temp2=self.config.temp2, temp3=self.config.temp3)
        local_pair_logits = mask_same_study_logits(local_match_outputs["pair_logits"], same_study_mask)
        local_gloria_loss = image_to_text_contrastive_loss(local_pair_logits, local_match_outputs["valid_mask"])
        local_gloria_loss = 0.5 * (local_gloria_loss + image_to_text_contrastive_loss(local_pair_logits.transpose(0, 1), local_match_outputs["valid_mask"]))

        local_aux_loss = images.sum() * 0.0
        if phrase_embeddings.shape[0] > 0 and phrase_positive_roi_mask.shape[0] == phrase_embeddings.shape[0]:
            if self.config.local_aux_type == "cosine":
                local_aux_loss = gloria_positive_cosine_loss(region_embeddings=region_embeddings, phrase_embeddings=phrase_embeddings,
                    phrase_batch_index=phrase_batch_index, phrase_positive_roi_mask=phrase_positive_roi_mask, roi_valid_mask=current_roi_valid_mask)
            else:
                local_aux_loss = gloria_attention_kl_loss(region_embeddings=region_embeddings, phrase_attention_embeddings=projected_phrase_embeddings,
                    phrase_similarity_embeddings=phrase_embeddings, phrase_positive_roi_mask=phrase_positive_roi_mask, phrase_batch_index=phrase_batch_index,
                    roi_valid_mask=current_roi_valid_mask, soft_target_alpha=self.config.attn_kl_soft_alpha)
        roi_last_hidden_state = current_roi_last_hidden_state
        roi_valid_mask = current_roi_valid_mask
        roi_boxes = current_roi_boxes

        # Keep global output fields compatible while the global path is disabled.
        global_sequence_length = frozen_outputs["global_backbone_tokens"].shape[1]
        global_height = int(math.sqrt(global_sequence_length))
        global_width = global_height
        global_height = global_height - (global_height % 2)
        global_width = global_width - (global_width % 2)
        if global_height <= 0 or global_width <= 0:
            num_global_patches = 1
        else:
            num_global_patches = (global_height // 2) * (global_width // 2)

        global_last_hidden_state = torch.zeros((batch_size, num_global_patches, self.visual_dim), dtype=roi_last_hidden_state.dtype, device=flat_images.device)
        temporal_global_last_hidden_state = global_last_hidden_state[:, None].expand(-1, temporal_length, -1, -1).clone()
        global_embeddings = torch.zeros((batch_size, self.text_embed_dim), dtype=roi_last_hidden_state.dtype, device=flat_images.device)
        global_pair_logits = torch.zeros((batch_size, batch_size), dtype=roi_last_hidden_state.dtype, device=flat_images.device)
        global_pair_logits = mask_same_study_logits(global_pair_logits, same_study_mask)
        global_loss = flat_images.sum() * 0.0
        temporal_class_logits = None
        # Abnormality training is disabled when cls_weight is 0, as set by the launcher.
        if self.config.cls_weight > 0.0:
            temporal_region_embeddings = self._project_region_embeddings(
                temporal_roi_last_hidden_state.reshape(batch_size * temporal_length, self.num_slots, num_roi_tokens, self.visual_dim),
                temporal_roi_valid_mask.reshape(batch_size * temporal_length, self.num_slots))
            flat_class_logits = self._compute_mil_logits(temporal_region_embeddings,
                temporal_roi_valid_mask.reshape(batch_size * temporal_length, self.num_slots))
            temporal_class_logits = flat_class_logits.reshape(batch_size, temporal_length, self.config.num_classes)
            class_logits = temporal_class_logits[:, -1]
            cls_loss = flat_images.sum() * 0.0
        else:
            class_logits = torch.zeros((batch_size, self.config.num_classes), device=flat_images.device)
            cls_loss = torch.tensor(0.0, device=flat_images.device)
        if temporal_image_labels is not None and temporal_image_label_mask is not None and self.config.cls_weight > 0.0:
            cls_loss = masked_multilabel_bce_per_image(flat_class_logits, temporal_image_labels.reshape(batch_size * temporal_length, self.config.num_classes),
                temporal_image_label_mask.reshape(batch_size * temporal_length, self.config.num_classes))
        elif image_labels is not None and image_label_mask is not None and self.config.cls_weight > 0.0:
            cls_loss = masked_multilabel_bce_per_image(class_logits, image_labels, image_label_mask)
        elif self.config.cls_weight == 0.0:
            cls_loss = torch.tensor(0.0, device=flat_images.device)

        loss = (self.config.cls_weight * cls_loss + self.config.local_gloria_weight * local_gloria_loss
            + self.config.local_aux_weight * local_aux_loss + self.config.global_contrastive_weight * global_loss)
        loss_dict = {"loss_cls": cls_loss, "loss_local_gloria": local_gloria_loss, "loss_local_aux": local_aux_loss, "loss_global": global_loss}

        if self.training:
            self.latest_loss_dict = {key: value.detach().float() for key, value in loss_dict.items()}

        if not return_dict:
            return (loss, loss_dict, local_pair_logits, global_pair_logits, region_embeddings, global_embeddings, roi_last_hidden_state,
                global_last_hidden_state, phrase_embeddings, class_logits, roi_boxes, roi_valid_mask)

        return GRPAOutput(loss=loss, loss_dict=loss_dict, local_pair_logits=local_pair_logits, global_pair_logits=global_pair_logits,
            region_embeddings=region_embeddings, global_embeddings=global_embeddings, roi_last_hidden_state=roi_last_hidden_state,
            global_last_hidden_state=global_last_hidden_state, text_embeddings=phrase_embeddings, class_logits=class_logits,
            temporal_class_logits=temporal_class_logits, pred_boxes=roi_boxes, roi_valid_mask=roi_valid_mask,
            temporal_roi_last_hidden_state=temporal_roi_last_hidden_state, temporal_global_last_hidden_state=temporal_global_last_hidden_state)
