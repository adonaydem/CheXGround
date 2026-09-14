import json
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict

from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
    Dinov2Model,
    Dinov2Config,
    DeformableDetrConfig
)
from transformers.utils import logging
from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrObjectDetectionOutput

from .ddetr_transformer import DeformableDetrTransformer

logger = logging.get_logger(__name__)


@dataclass
class CustomDDETRModelOutput(DeformableDetrObjectDetectionOutput):
    logits: Optional[Dict[str, torch.FloatTensor]] = None
    pred_boxes: Optional[Dict[str, torch.FloatTensor]] = None


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CustomDDETRConfig(PretrainedConfig):
    model_type = "ddetr"

    def __init__(
        self,
        vis_encoder_cfg=None,
        zs_weight_path=None,
        vis_output_layer=-1,
        ddetr_cfg=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vis_encoder_cfg is None:
            self.vis_encoder_cfg = Dinov2Config()
            logger.info("vis_encoder_config is None. initializing the Dinov2Config with default values.")
        elif isinstance(vis_encoder_cfg, dict):
            self.vis_encoder_cfg = Dinov2Config(**vis_encoder_cfg)
        elif isinstance(vis_encoder_cfg, Dinov2Config):
            self.vis_encoder_cfg = vis_encoder_cfg
        else:
            raise NotImplementedError("currently only supports Dinov2Model as vis_encoder.")

        if ddetr_cfg is None:
            self.ddetr_cfg = DeformableDetrConfig()
            logger.info("ddetr_cfg is None. Initializing the DeformableDetr with default values.")
        elif isinstance(ddetr_cfg, dict):
            self.ddetr_cfg = DeformableDetrConfig(**ddetr_cfg)
        elif isinstance(ddetr_cfg, DeformableDetrConfig):
            self.ddetr_cfg = ddetr_cfg
        else:
            raise NotImplementedError("currently only supports DeformableDetrTransformer as detector head.")

        self.zs_weight_path = zs_weight_path
        self.vis_output_layer = vis_output_layer

    def to_json_string(self, use_diff: bool = True) -> str:
        if use_diff is True:
            config_dict = copy.deepcopy(self)
            config_dict.vis_encoder_cfg = config_dict.vis_encoder_cfg.to_diff_dict()
            config_dict.ddetr_cfg = config_dict.ddetr_cfg.to_diff_dict()
            config_dict = config_dict.to_diff_dict()
        else:
            config_dict = copy.deepcopy(self)
            config_dict.vis_encoder_cfg = config_dict.vis_encoder_cfg.to_dict()
            config_dict.ddetr_cfg = config_dict.ddetr_cfg.to_dict()
            config_dict = config_dict.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"


class CustomDDETRModel(PreTrainedModel):
    config_class = CustomDDETRConfig

    def __init__(self, config: CustomDDETRConfig, pretrained_vis_encoder=None):
        super().__init__(config)

        if pretrained_vis_encoder is not None:
            self.vis_encoder = Dinov2Model.from_pretrained(pretrained_vis_encoder)
        else:
            self.vis_encoder = Dinov2Model(config.vis_encoder_cfg)

        self.ddetr_transformer = DeformableDetrTransformer(config.ddetr_cfg, config.zs_weight_path)
        self.region_num_levels = config.ddetr_cfg.num_feature_levels
        self._vis_encoder_frozen = False
        self.train_bbox = bool(getattr(config.ddetr_cfg, "train_bbox", True))
        num_feature_levels = config.ddetr_cfg.num_feature_levels
        in_channels = config.vis_encoder_cfg.hidden_size
        input_proj_list = []
        if num_feature_levels == 1:
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, config.ddetr_cfg.d_model, kernel_size=1),
                LayerNorm(config.ddetr_cfg.d_model),
            ))
        elif num_feature_levels == 3:
            input_proj_list.extend([
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, config.ddetr_cfg.d_model // 2, kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.Conv2d(config.ddetr_cfg.d_model // 2, config.ddetr_cfg.d_model, kernel_size=1),
                    LayerNorm(config.ddetr_cfg.d_model),
                    nn.Conv2d(config.ddetr_cfg.d_model, config.ddetr_cfg.d_model, kernel_size=3, padding=1),
                    LayerNorm(config.ddetr_cfg.d_model),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, config.ddetr_cfg.d_model, kernel_size=1),
                    LayerNorm(config.ddetr_cfg.d_model),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, config.ddetr_cfg.d_model, kernel_size=3, stride=2, padding=1),
                    LayerNorm(config.ddetr_cfg.d_model),
                    nn.Conv2d(config.ddetr_cfg.d_model, config.ddetr_cfg.d_model, kernel_size=3, padding=1),
                    LayerNorm(config.ddetr_cfg.d_model),
                ),
            ])
        else:
            raise ValueError("CustomDDETRModel currently supports num_feature_levels in {1, 3}.")
        self.input_proj = nn.ModuleList(input_proj_list)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def freeze_vis_encoder(self):
        self.vis_encoder.requires_grad_(False)
        self._vis_encoder_frozen = True

    def freeze_ddetr(self):
        self.ddetr_transformer.requires_grad_(False)

    def get_vis_encoder(self):
        return getattr(self, 'vis_encoder', None)

    def get_ddetr(self):
        return getattr(self, 'ddetr_transformer', None)

    def forward(
        self,
        images: Optional[list] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CustomDDETRModelOutput]:
        if self._vis_encoder_frozen:
            with torch.no_grad():
                image_forward_outs = self.vis_encoder(images, output_hidden_states=True)
        else:
            image_forward_outs = self.vis_encoder(images, output_hidden_states=True)

        if self.region_num_levels == 1:
            # CheXGround's ordering.
            select_hidden_state = torch.stack(image_forward_outs.hidden_states[-4:])
            image_features = torch.mean(select_hidden_state, dim=0)
            image_features = image_features[:, 1:]
            batch_size, sequence_length, hidden_dim = image_features.shape
            height = width = int(math.sqrt(sequence_length))
            if height * width != sequence_length:
                raise ValueError(f"Expected square token grid, got sequence_length={sequence_length}.")
            backbone_feature_maps = [
                image_features.reshape(batch_size, height, width, hidden_dim).permute(0, 3, 1, 2).contiguous()
            ]
        elif self.region_num_levels == 3:
            backbone_feature_maps = []
            for hidden_state in image_forward_outs.hidden_states[-3:]:
                hidden_state = hidden_state[:, 1:]
                batch_size, sequence_length, hidden_dim = hidden_state.shape
                height = width = int(math.sqrt(sequence_length))
                if height * width != sequence_length:
                    raise ValueError(f"Expected square token grid, got sequence_length={sequence_length}.")
                backbone_feature_maps.append(
                    hidden_state.reshape(batch_size, height, width, hidden_dim).permute(0, 3, 1, 2).contiguous()
                )
        else:
            raise ValueError(f"Unsupported region_num_levels={self.region_num_levels}.")

        srcs = [input_proj(feature_map) for input_proj, feature_map in zip(self.input_proj, backbone_feature_maps)]
        if pixel_mask is None:
            pixel_mask = torch.ones(
                (images.shape[0], images.shape[-2], images.shape[-1]),
                dtype=torch.bool,
                device=images.device,
            )
        else:
            pixel_mask = pixel_mask.to(device=images.device, dtype=torch.bool)
        if tuple(pixel_mask.shape) != (images.shape[0], images.shape[-2], images.shape[-1]):
            raise ValueError(
                "pixel_mask must have shape [batch, height, width] aligned with images. "
                f"Got pixel_mask={tuple(pixel_mask.shape)} images={tuple(images.shape)}."
            )
        masks = [
            F.interpolate(pixel_mask[:, None].float(), size=src.shape[-2:], mode="nearest")[:, 0] > 0.5
            for src in srcs
        ]

        ddetr_outputs = self.ddetr_transformer(
            sources=srcs,
            masks=masks,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if self.train_bbox:
            loss = ddetr_outputs.loss
            loss_dict = dict(ddetr_outputs.loss_dict or {}) if ddetr_outputs.loss_dict is not None else {}
        else:
            loss = 0.0
            if hasattr(ddetr_outputs, "pred_boxes") and ddetr_outputs.pred_boxes.get("anatomy") is not None:
                loss += torch.sum(ddetr_outputs.pred_boxes["anatomy"]) * 0.0
            
            # preserve original shapes
            for param in self.ddetr_transformer.parameters():
                 if param.requires_grad:
                     loss += torch.sum(param) * 0.0
            
            loss_dict = {}

        output_logits = ddetr_outputs.logits
        output_boxes = ddetr_outputs.pred_boxes

        if self.training and loss_dict is not None:
            self.latest_loss_dict = {
                k: v.detach().cpu().item() for k, v in loss_dict.items() if isinstance(v, torch.Tensor)
            }

        if not (return_dict if return_dict is not None else self.config.return_dict):
            output = (output_logits, output_boxes)
            return ((loss, loss_dict) + output) if loss is not None else output

        return CustomDDETRModelOutput(
            loss=loss,
            loss_dict=loss_dict or None,
            logits=output_logits,
            pred_boxes=output_boxes,
            auxiliary_outputs=ddetr_outputs.auxiliary_outputs,
            last_hidden_state=ddetr_outputs.last_hidden_state,
            decoder_hidden_states=ddetr_outputs.decoder_hidden_states,
            decoder_attentions=ddetr_outputs.decoder_attentions,
            cross_attentions=ddetr_outputs.cross_attentions,
            encoder_last_hidden_state=ddetr_outputs.encoder_last_hidden_state,
            encoder_hidden_states=ddetr_outputs.encoder_hidden_states,
            encoder_attentions=ddetr_outputs.encoder_attentions,
            intermediate_hidden_states=ddetr_outputs.intermediate_hidden_states,
            intermediate_reference_points=ddetr_outputs.intermediate_reference_points,
            init_reference_points=ddetr_outputs.init_reference_points,
            enc_outputs_class=ddetr_outputs.enc_outputs_class,
            enc_outputs_coord_logits=ddetr_outputs.enc_outputs_coord_logits,
        )


AutoConfig.register("ddetr", CustomDDETRConfig)
AutoModel.register(CustomDDETRConfig, CustomDDETRModel)
