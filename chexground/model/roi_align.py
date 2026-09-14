import re
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, Linear, normal_init
from mmdet.models import BaseRoIExtractor
from transformers.image_transforms import center_to_corners_format


def str2spi(input_str):
    bbox_regex = r'<bbox>\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)\s*</bbox>'
    results = []
    matches = re.findall(bbox_regex, input_str)
    for match in matches:
        results.append([float(match[0]), float(match[1]), float(match[2]),
                        float(match[3])])
    return results


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def coordinate_to_encoding(coord_tensor,
                           num_feats: int = 128,
                           temperature: int = 10000,
                           scale: float = 2 * math.pi):
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=coord_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(2)
    if coord_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)
    elif coord_tensor.size(-1) == 4:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                            dim=-1).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
                            dim=-1).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
            coord_tensor.size(-1)))
    return pos


def align_tensor(inputs, max_len=None):
    if max_len is None:
        max_len = max([len(item) for item in inputs])

    return torch.stack([padding_to(item, max_len) for item in inputs])


def padding_to(inputs, max=300):
    if max is None:
        return inputs
    num_padding = max - len(inputs)
    if inputs.dim() > 1:
        padding = inputs.new_zeros(num_padding,
                                   *inputs.size()[1:],
                                   dtype=inputs.dtype)
    else:
        padding = inputs.new_zeros(num_padding, dtype=inputs.dtype)
    inputs = torch.cat([inputs, padding], dim=0)
    return inputs


class MLVLFuseModule(nn.Module):
    def __init__(self, input_dims=1024, embed_dims=1024, num_levels=3, num_fuse=4):
        super(MLVLFuseModule, self).__init__()
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_fuse = num_fuse
        self.input_dims = input_dims
        self.shuffle_channles = embed_dims // 4

        # contains the tuple of level indices that will do the interaction
        self.fuse_lvl_list = []
        num_levels = self.num_levels
        for lvl in range(num_levels):
            top_lvl = min(lvl + 1, num_levels - 1)
            dow_lvl = max(lvl - 1, 0)
            tar_lvl = lvl
            self.fuse_lvl_list.append((tar_lvl, top_lvl, dow_lvl))

        self.remain_chs = self.embed_dims - self.shuffle_channles * 2
        self._init_layers()

    def generate_coordinate(self, featmap_sizes, device='cuda'):
        x_range = torch.linspace(-1, 1, featmap_sizes[-1], device=device)
        y_range = torch.linspace(-1, 1, featmap_sizes[-2], device=device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([featmap_sizes[0], 1, -1, -1])
        x = x.expand([featmap_sizes[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        return coord_feat

    def _init_layers(self):
        self.input_conv = nn.ModuleList(
            [nn.Conv2d(self.input_dims + 2, self.embed_dims, 1) for _ in range(self.num_levels)])
        self.fuse_convs = nn.ModuleList()
        for i in range(self.num_fuse):
            self.fuse_convs.append(
                ConvModule(self.embed_dims,
                           self.embed_dims,
                           3,
                           stride=1,
                           padding=3 // 2,
                           conv_cfg=None,
                           norm_cfg=dict(type='GN',
                                         num_groups=64,
                                         requires_grad=True)
                           ))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def _single_shuffle(self, inputs, conv_module):
        if not isinstance(conv_module, (nn.ModuleList, list)):
            conv_module = [conv_module]
        for single_conv_m in conv_module:
            fused_inputs = []
            for fuse_lvl_tuple in self.fuse_lvl_list:
                tar_lvl, top_lvl, dow_lvl = fuse_lvl_tuple
                tar_input = inputs[tar_lvl]
                top_input = inputs[top_lvl]
                down_input = inputs[dow_lvl]
                remain = tar_input[:, :self.remain_chs]
                from_top = top_input[:,
                           self.remain_chs:][:,
                           self.shuffle_channles:]
                from_top = F.interpolate(from_top.to(torch.float32),
                                         size=tar_input.shape[-2:],
                                         mode='bilinear',
                                         align_corners=True)
                from_down = down_input[:, self.remain_chs:][:, :self.
                shuffle_channles]
                from_down = F.interpolate(from_down.to(torch.float32),
                                          size=tar_input.shape[-2:],
                                          mode='bilinear',
                                          align_corners=True)
                fused_inputs.append(
                    torch.cat([remain, from_top, from_down], dim=1))
            fused_inputs = [single_conv_m(item) for item in fused_inputs]
            inputs = fused_inputs
        return inputs

    def forward(self, inputs):
        feat_size = [item.shape for item in inputs]
        new_inputs = []
        for feat, single_feat_size in zip(inputs, feat_size):
            coord_feat = self.generate_coordinate(single_feat_size, device=inputs[0].device)
            feat = torch.cat([feat, coord_feat], dim=1)
            new_inputs.append(feat)
        inputs = new_inputs

        inputs = [self.input_conv[lvl](item) for lvl, item in enumerate(inputs)]

        for conv_m in self.fuse_convs:
            inputs = self._single_shuffle(inputs, [conv_m])
        return inputs


class MLVLROIQueryModule(nn.Module):
    def __init__(
        self,
        embed_dims=1024,
        out_dims=4096,
        num_levels=3,
        return_spatial=False,
        roi_output_size=7,
        roi_sampling_ratio=2,
    ):
        super(MLVLROIQueryModule, self).__init__()
        self.mlvl_fuse = MLVLFuseModule(
            input_dims=embed_dims,
            embed_dims=embed_dims,
            num_levels=num_levels,
            num_fuse=5)
        bbox_roi_extractor = dict(
            # RoIs are scaled to each feature level explicitly in forward, so
            # the underlying ops can use unit spatial scale regardless of the
            # input image size or feature pyramid geometry.
            roi_layer=dict(type='RoIAlign', output_size=roi_output_size, sampling_ratio=roi_sampling_ratio),
            out_channels=embed_dims,
            embed_dims=embed_dims,
            fuse_level=num_levels,
            featmap_strides=[1.0 for _ in range(num_levels)],
            output_dims=out_dims,
            return_spatial=return_spatial,
        )

        self.roi_align = MlvlRoIExtractor(**bbox_roi_extractor)

    def forward(self, mlvl_feats, bboxes, image_shapes=None):
        if mlvl_feats[0].dim() == 3:
            sequence_length = mlvl_feats[0].shape[1]
            h = w = int(math.sqrt(sequence_length))
            if h * w != sequence_length:
                raise ValueError(
                    f"Expected square token grid for RoI features, got sequence_length={sequence_length}."
                )
            b, c = mlvl_feats[0].shape[0], mlvl_feats[0].shape[-1]
            mlvl_feats = [item.reshape(b, h, w, c).permute(0, 3, 1, 2) for item in mlvl_feats]
        mlvl_feats = self.mlvl_fuse(mlvl_feats)

        return self.roi_align(mlvl_feats, bboxes, image_shapes=image_shapes)


class MlvlRoIExtractor(BaseRoIExtractor):
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 embed_dims=1024,
                 output_dims=1024,
                 return_spatial=False,
                 stride=1,
                 norm_init=True,
                 fuse_level=3,
                 finest_scale=56,
                 init_cfg=None):
        super(MlvlRoIExtractor, self).__init__(roi_layer, out_channels,
                                               featmap_strides, init_cfg)
        self.embed_dims = embed_dims
        self.output_dims = output_dims
        self.return_spatial = return_spatial
        self.finest_scale = finest_scale
        self.fuse_level = fuse_level
        self.norm_init = norm_init

        self.pconvs = nn.ModuleList(
            nn.Conv2d(self.embed_dims, self.embed_dims, 3, stride=1, padding=1)
            for _ in range(self.fuse_level))
        
        if not self.return_spatial:
            self.pos_embedd = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.LayerNorm(1024),
            )
            self.updims = nn.Linear(1024, self.output_dims)

            output_size = self.roi_layers[0].output_size
            if isinstance(output_size, int):
                output_height = output_width = output_size
            else:
                output_height, output_width = output_size
            self.flatten_linear = nn.Linear(self.embed_dims * output_height * output_width, 1024)
        else:
            self.updims = None
            self.flatten_linear = None

        self.norm_init_weights()

    def norm_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)

    @staticmethod
    def _normalized_rois_to_feature_rois(single_img_roi, img_id, feat_height, feat_width):
        feature_rois = center_to_corners_format(single_img_roi).clamp(min=0.0, max=1.0)
        feature_rois = feature_rois.clone()
        feature_rois[:, 0::2] *= feat_width
        feature_rois[:, 1::2] *= feat_height
        roi_img_id = feature_rois.new_full((len(feature_rois), 1), float(img_id))
        return torch.cat([roi_img_id, feature_rois], dim=1)

    def forward(self, feats, rois, roi_scale_factor=None, image_shapes=None):
        """Forward function."""
        del roi_scale_factor, image_shapes
        num_imgs = len(rois)
        batched_rois = torch.cat(rois, dim=0)
        
        out_size = self.roi_layers[0].output_size
        if isinstance(out_size, int):
            out_size = (out_size, out_size)
        num_levels = len(feats)
        if feats[0].dim() == 3:
            sequence_length = feats[0].shape[1]
            h = w = int(math.sqrt(sequence_length))
            if h * w != sequence_length:
                raise ValueError(
                    f"Expected square token grid for RoI features, got sequence_length={sequence_length}."
                )
            b, c = feats[0].shape[0], feats[0].shape[-1]
            feats = [item.reshape(b, h, w, c).permute(0, 3, 1, 2) for item in feats]
        roi_img_ids = torch.cat(
            [single_img_roi.new_full((len(single_img_roi),), img_id, dtype=torch.long) for img_id, single_img_roi in enumerate(rois)],
            dim=0,
        )

        roi_feats = feats[0].new_zeros(self.fuse_level,
                                       batched_rois.size(0), self.out_channels, *out_size)

        for i in range(num_levels):
            if batched_rois.shape[0] > 0:
                feat_height, feat_width = feats[i].shape[-2:]
                rois_ = torch.cat(
                    [
                        self._normalized_rois_to_feature_rois(single_img_roi, img_id, feat_height, feat_width)
                        for img_id, single_img_roi in enumerate(rois)
                    ],
                    dim=0,
                )
                ori_dtype = feats[i].dtype
                roi_feats_t = self.roi_layers[i](feats[i].to(torch.float32), rois_.to(torch.float32))

                roi_feats[i] = roi_feats_t.to(ori_dtype)

            else:
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.

        per_level_roi_feats = []
        for i in range(self.fuse_level):
            per_level_roi_feats.append(F.relu(self.pconvs[i](roi_feats[i])))

        query_feats = []
        for i in range(num_imgs):
            mask = roi_img_ids == i
            if self.return_spatial:
                query_feats.append(torch.stack([feat[mask] for feat in per_level_roi_feats], dim=1))
        # [num_rois, num_levels, C, H, W]
        if self.return_spatial:
            return query_feats
        pos_embedd = self.pos_embedd(batched_rois)
        fuse_roi_feats = sum(per_level_roi_feats)
        fuse_roi_feats = fuse_roi_feats.flatten(1, -1)
        fuse_roi_feats = self.flatten_linear(fuse_roi_feats)
        fuse_roi_feats = fuse_roi_feats + pos_embedd
        fuse_roi_feats = self.updims(fuse_roi_feats)
        query_feats = []
        for i in range(num_imgs):
            mask = roi_img_ids == i
            query_feats.append(fuse_roi_feats[mask])

        return query_feats
