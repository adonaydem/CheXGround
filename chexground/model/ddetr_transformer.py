import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from torch.nn.init import constant_, normal_, xavier_uniform_

from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrConfig,
    DeformableDetrDecoder,
    DeformableDetrDecoderLayer,
    DeformableDetrDecoderOutput,
    DeformableDetrEncoder,
    DeformableDetrMLPPredictionHead,
    DeformableDetrMultiscaleDeformableAttention,
    DeformableDetrObjectDetectionOutput,
    DeformableDetrPreTrainedModel,
    _get_clones,
    build_position_encoding,
    inverse_sigmoid,
)
try:
    from torchvision.ops import generalized_box_iou
except ImportError:
    from transformers.image_transforms import generalized_box_iou

def _generalized_box_iou(boxes1, boxes2):
    return generalized_box_iou(boxes1, boxes2)
from transformers.image_transforms import center_to_corners_format


@dataclass
class MedDeformableDetrObjectDetectionOutput(DeformableDetrObjectDetectionOutput):
    logits: Optional[Dict[str, torch.FloatTensor]] = None
    pred_boxes: Optional[Dict[str, torch.FloatTensor]] = None
    spatial_shapes: Optional[torch.LongTensor] = None
    level_start_index: Optional[torch.LongTensor] = None
    encoder_feature_maps: Optional[Tuple[torch.FloatTensor, ...]] = None


class DeformableDetrDecoderX(DeformableDetrDecoder):
    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layers = nn.ModuleList([DeformableDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.gradient_checkpointing = False
        self.bbox_embed = None
        self.post_init()

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()

        for idx, decoder_layer in enumerate(self.layers):
            num_coordinates = reference_points.shape[-1]
            if num_coordinates == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                if num_coordinates != 2:
                    raise ValueError("Reference points' last dimension must be of size 2")
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(
                        hidden_states,
                        position_embeddings,
                        encoder_hidden_states,
                        reference_points_input,
                        spatial_shapes,
                        spatial_shapes_list,
                        level_start_index,
                        encoder_attention_mask,
                    ):
                        layer_kwargs = {
                            "position_embeddings": position_embeddings,
                            "encoder_hidden_states": encoder_hidden_states,
                            "reference_points": reference_points_input,
                            "spatial_shapes": spatial_shapes,
                            "level_start_index": level_start_index,
                            "encoder_attention_mask": encoder_attention_mask,
                            "output_attentions": output_attentions,
                        }
                        return module(
                            hidden_states,
                            **layer_kwargs,
                            spatial_shapes_list=spatial_shapes_list,
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    position_embeddings,
                    encoder_hidden_states,
                    reference_points_input,
                    spatial_shapes,
                    spatial_shapes_list,
                    level_start_index,
                    encoder_attention_mask,
                )
            else:
                layer_kwargs = {
                    "position_embeddings": position_embeddings,
                    "encoder_hidden_states": encoder_hidden_states,
                    "reference_points": reference_points_input,
                    "spatial_shapes": spatial_shapes,
                    "level_start_index": level_start_index,
                    "encoder_attention_mask": encoder_attention_mask,
                    "output_attentions": output_attentions,
                }
                layer_outputs = decoder_layer(
                    hidden_states,
                    **layer_kwargs,
                    spatial_shapes_list=spatial_shapes_list,
                )

            hidden_states = layer_outputs[0]

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                if num_coordinates == 4:
                    reference_points = (tmp + inverse_sigmoid(reference_points)).sigmoid().detach()
                else:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    reference_points = new_reference_points.sigmoid().detach()

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return DeformableDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class DeformableDetrTransformer(DeformableDetrPreTrainedModel):
    def __init__(self, config: DeformableDetrConfig, zs_weight_path=None):
        super().__init__(config)

        anatomy_num_queries = getattr(config, "anatomy_num_queries", config.num_queries)
        config.anatomy_num_queries = anatomy_num_queries
        self.anatomy_num_queries = anatomy_num_queries
        
        if anatomy_num_queries <= 0:
            raise ValueError(f"anatomy_num_queries must be positive, got {anatomy_num_queries}.")
        
        
        if config.two_stage:
            raise NotImplementedError("two-stage Deformable DETR is not supported for anatomy-only detection.")

        self.encoder = DeformableDetrEncoder(config)
        self.decoder_anatomy = DeformableDetrDecoderX(config)
        self.position_encoding = build_position_encoding(config)
        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))
        self.query_position_embeddings_anatomy = nn.Embedding(self.anatomy_num_queries, config.d_model * 2)
        self.reference_points_anatomy = nn.Linear(config.d_model, 2)

        self._reset_parameters()

        num_pred = config.decoder_layers
        bbox_embed = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=256,
            output_dim=4,
            num_layers=3,
        )
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        if config.with_box_refine:
            bbox_embed = _get_clones(bbox_embed, num_pred)
            nn.init.constant_(bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.decoder_anatomy.bbox_embed = bbox_embed
        else:
            nn.init.constant_(bbox_embed.layers[-1].bias.data[2:], -2.0)
            bbox_embed = nn.ModuleList([bbox_embed for _ in range(num_pred)])
            self.decoder_anatomy.bbox_embed = None
        self.bbox_embed_anatomy = bbox_embed

    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for module in self.modules():
            if isinstance(module, DeformableDetrMultiscaleDeformableAttention):
                if hasattr(module, "_reset_parameters"):
                    module._reset_parameters()
                else:
                    if hasattr(module, "sampling_offsets"):
                        constant_(module.sampling_offsets.weight.data, 0.0)
                        if module.sampling_offsets.bias is not None:
                            the_heads = getattr(module, "n_heads", 8)
                            the_levels = getattr(module, "n_levels", 4)
                            the_points = getattr(module, "n_points", 4)
                            grid_init = torch.stack([
                                torch.linspace(-1, 1, the_points) for _ in range(the_heads)
                            ], dim=0).unsqueeze(1).unsqueeze(-1)
                            grid_init = grid_init.expand(the_heads, the_levels, the_points, 2)
                            grid_init_flat = grid_init.reshape(-1).to(device=module.sampling_offsets.bias.device, dtype=module.sampling_offsets.bias.dtype)
                            if list(module.sampling_offsets.bias.data.shape) == list(grid_init_flat.shape):
                                module.sampling_offsets.bias.data.copy_(grid_init_flat)
                            else:
                                constant_(module.sampling_offsets.bias.data, 0.0)
                    if hasattr(module, "attention_weights"):
                        constant_(module.attention_weights.weight.data, 0.0)
                        if module.attention_weights.bias is not None:
                            constant_(module.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.reference_points_anatomy.weight.data, gain=1.0)
        constant_(self.reference_points_anatomy.bias.data, 0.0)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        return torch.stack([valid_width.float() / width, valid_height.float() / height], -1)

    def _normalize_labels(self, labels, device):
        anatomy_targets = []
        for label in labels:
            anatomy_target = (label or {}).get("anatomy") or {}
            anatomy_boxes = anatomy_target.get("boxes")
            if anatomy_boxes is None:
                anatomy_boxes = torch.zeros((self.anatomy_num_queries, 4), dtype=torch.float32, device=device)
            else:
                anatomy_boxes = torch.as_tensor(anatomy_boxes, dtype=torch.float32, device=device)

            slot_mask = anatomy_target.get("slot_mask")
            if slot_mask is None:
                slot_mask = torch.zeros((self.anatomy_num_queries,), dtype=torch.bool, device=device)
            else:
                slot_mask = torch.as_tensor(slot_mask, dtype=torch.bool, device=device)

            anatomy_targets.append({"boxes": anatomy_boxes, "slot_mask": slot_mask})
        return anatomy_targets

    def _build_decoder_inputs(self, batch_size, num_channels):
        query_embeds = self.query_position_embeddings_anatomy.weight
        query_embed, target = torch.split(query_embeds, num_channels, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        target = target.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = self.reference_points_anatomy(query_embed).sigmoid()
        return query_embed, target, reference_points

    def _compute_box_outputs(self, hidden_states, inter_references, init_reference):
        outputs_coord = []
        for level in range(hidden_states.shape[1]):
            reference = init_reference if level == 0 else inter_references[:, level - 1]
            reference = inverse_sigmoid(reference)
            delta_bbox = self.bbox_embed_anatomy[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                outputs_coord_logits = delta_bbox + reference
            elif reference.shape[-1] == 2:
                delta_bbox = delta_bbox.clone()
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}")
            outputs_coord.append(outputs_coord_logits.sigmoid())
        return torch.stack(outputs_coord, dim=1)

    def _reshape_encoder_feature_maps(self, encoder_last_hidden_state, spatial_shapes, level_start_index):
        encoder_feature_maps = []
        batch_size, _, num_channels = encoder_last_hidden_state.shape
        for level, (height, width) in enumerate(spatial_shapes.tolist()):
            start = level_start_index[level].item()
            end = start + height * width
            level_features = encoder_last_hidden_state[:, start:end]
            level_features = level_features.transpose(1, 2).reshape(batch_size, num_channels, height, width)
            encoder_feature_maps.append(level_features)
        return tuple(encoder_feature_maps)

    def _build_loss_weight_dict(self):
        weight_dict = {
            "loss_bbox": self.config.bbox_loss_coefficient,
            "loss_giou": self.config.giou_loss_coefficient,
        }
        if self.config.auxiliary_loss:
            aux_weight_dict = {}
            for i in range(self.config.decoder_layers - 1):
                aux_weight_dict.update({f"{k}_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        return weight_dict

    def _compute_anatomy_box_losses(self, outputs_coord, anatomy_targets):
        losses = {}
        anatomy_boxes = torch.stack([target["boxes"] for target in anatomy_targets], dim=0)
        anatomy_slot_mask = torch.stack([target["slot_mask"] for target in anatomy_targets], dim=0)
        if anatomy_boxes.shape[1] != self.anatomy_num_queries:
            raise ValueError(
                f"Anatomy boxes must align to anatomy_num_queries={self.anatomy_num_queries}, "
                f"got {anatomy_boxes.shape[1]}."
            )
        if anatomy_slot_mask.shape != anatomy_boxes.shape[:2]:
            raise ValueError("Anatomy slot_mask must have shape [batch_size, anatomy_num_queries].")

        num_valid = anatomy_slot_mask.sum()
        if num_valid.item() == 0:
            return losses

        norm = num_valid.float()
        pred_boxes = outputs_coord[:, -1]
        valid_pred_boxes = pred_boxes[anatomy_slot_mask]
        valid_target_boxes = anatomy_boxes[anatomy_slot_mask]
        losses["loss_bbox"] = nn.functional.l1_loss(valid_pred_boxes, valid_target_boxes, reduction="none").sum() / norm
        losses["loss_giou"] = (
            1
            - torch.diag(
                generalized_box_iou(
                    center_to_corners_format(valid_pred_boxes),
                    center_to_corners_format(valid_target_boxes),
                )
            )
        ).sum() / norm

        if self.config.auxiliary_loss:
            for i, aux_pred_boxes in enumerate(outputs_coord[:, :-1].unbind(dim=1)):
                valid_aux_boxes = aux_pred_boxes[anatomy_slot_mask]
                losses[f"loss_bbox_{i}"] = (
                    nn.functional.l1_loss(valid_aux_boxes, valid_target_boxes, reduction="none").sum() / norm
                )
                losses[f"loss_giou_{i}"] = (
                    1
                    - torch.diag(
                        generalized_box_iou(
                            center_to_corners_format(valid_aux_boxes),
                            center_to_corners_format(valid_target_boxes),
                        )
                    )
                ).sum() / norm

        return losses

    def extract_feature(
        self,
        sources,
        masks,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        position_embeddings_list = []
        for source, mask in zip(sources, masks):
            position_embeddings_list.append(self.position_encoding(source, mask).to(source.dtype))

        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
            batch_size, _, height, width = source.shape
            spatial_shapes.append((height, width))
            source_flatten.append(source.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed_flatten.append(pos_embed + self.level_embed[level].view(1, 1, -1))

        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        spatial_shapes_list = [tuple(level_shape) for level_shape in spatial_shapes.tolist()]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(mask) for mask in masks], 1).float()

        encoder_kwargs = {
            "inputs_embeds": source_flatten,
            "attention_mask": mask_flatten,
            "position_embeddings": lvl_pos_embed_flatten,
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "valid_ratios": valid_ratios,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        encoder_outputs = self.encoder(
            **encoder_kwargs,
            spatial_shapes_list=spatial_shapes_list,
        )

        batch_size, _, num_channels = encoder_outputs.last_hidden_state.shape
        query_embed, target, reference_points = self._build_decoder_inputs(batch_size, num_channels)
        decoder_outputs = self.decoder_anatomy(
            inputs_embeds=target,
            position_embeddings=query_embed,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        return {
            "init_reference_points": reference_points,
            "last_hidden_state": decoder_outputs.last_hidden_state,
            "intermediate_hidden_states": decoder_outputs.intermediate_hidden_states,
            "intermediate_reference_points": decoder_outputs.intermediate_reference_points,
            "decoder_hidden_states": decoder_outputs.hidden_states,
            "decoder_attentions": decoder_outputs.attentions,
            "cross_attentions": decoder_outputs.cross_attentions,
            "encoder_last_hidden_state": encoder_outputs.last_hidden_state,
            "encoder_hidden_states": encoder_outputs.hidden_states,
            "encoder_attentions": encoder_outputs.attentions,
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "encoder_feature_maps": self._reshape_encoder_feature_maps(
                encoder_outputs.last_hidden_state,
                spatial_shapes,
                level_start_index,
            ),
        }

    def forward(
        self,
        sources,
        masks,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        outputs = self.extract_feature(
            sources=sources,
            masks=masks,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        anatomy_outputs_coord = self._compute_box_outputs(
            outputs["intermediate_hidden_states"],
            outputs["intermediate_reference_points"],
            outputs["init_reference_points"],
        )
        pred_boxes = {"anatomy": anatomy_outputs_coord[:, -1]}
        logits = {}

        loss = None
        loss_dict = None
        if labels is not None:
            anatomy_targets = self._normalize_labels(labels, sources[0].device)
            weight_dict = self._build_loss_weight_dict()
            anatomy_loss_dict = self._compute_anatomy_box_losses(anatomy_outputs_coord, anatomy_targets)
            loss = sources[0].sum() * 0.0
            loss_dict = {}
            for loss_name, loss_value in anatomy_loss_dict.items():
                prefixed_name = f"anatomy_{loss_name}"
                loss_dict[prefixed_name] = loss_value
                if loss_name in weight_dict:
                    loss = loss + loss_value * weight_dict[loss_name]

        if not return_dict:
            output = (logits, pred_boxes, outputs)
            return ((loss, loss_dict) + output) if loss is not None else output

        return MedDeformableDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=None,
            last_hidden_state=outputs["last_hidden_state"],
            decoder_hidden_states=outputs["decoder_hidden_states"],
            decoder_attentions=outputs["decoder_attentions"],
            cross_attentions=outputs["cross_attentions"],
            encoder_last_hidden_state=outputs["encoder_last_hidden_state"],
            encoder_hidden_states=outputs["encoder_hidden_states"],
            encoder_attentions=outputs["encoder_attentions"],
            intermediate_hidden_states=outputs["intermediate_hidden_states"],
            intermediate_reference_points=outputs["intermediate_reference_points"],
            init_reference_points=outputs["init_reference_points"],
            enc_outputs_class=None,
            enc_outputs_coord_logits=None,
            spatial_shapes=outputs["spatial_shapes"],
            level_start_index=outputs["level_start_index"],
            encoder_feature_maps=outputs["encoder_feature_maps"],
        )
