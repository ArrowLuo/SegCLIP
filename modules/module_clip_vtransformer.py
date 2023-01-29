"""
Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from modules.module_seg_vit import SegViT
from modules.module_clip_util import LayerNorm, QuickGELU, random_masking

class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, first_stage_layer: int=10):

        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))

        self.ln_pre = LayerNorm(width)
        self.transformer = SegViT(width, patch_size=patch_size, input_resolution=input_resolution, first_stage_layer=first_stage_layer)

        self.ln_post = LayerNorm(width)

        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def get_pos_embed(self, h_, w_):
        if self.training:
            return self.positional_embedding
        pos_embed = self.positional_embedding
        pos_cls, pos_embed = torch.split(pos_embed, (1, pos_embed.size(0)-1), dim=0)

        # interpolate pos encoding
        num_patches = h_ * w_
        n, dim = pos_embed.size()
        if num_patches == n and w_ == h_:
            return torch.cat([pos_cls, pos_embed], dim=0)

        patch_pos_embed = F.interpolate(pos_embed.reshape(1, int(math.sqrt(n)), int(math.sqrt(n)), dim).permute(0, 3, 1, 2),
                                        size=(h_, w_), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)

        patch_pos_embed = torch.cat([pos_cls, patch_pos_embed], dim=0)

        return patch_pos_embed

    def forward(self, x: torch.Tensor, video_frame=-1, mask_ratio=0.):
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        h_, w_ = x.size(-2), x.size(-1)

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + self.get_pos_embed(h_, w_).to(x.dtype)
        x = self.ln_pre(x)

        # MAE ===>
        mae_mask, mae_ids_restore, x_mask, ids_keep = None, None, None, None
        if mask_ratio > 0.:
            # masking: length -> length * mask_ratio
            x, mae_mask, mae_ids_restore, ids_keep = random_masking(x, mask_ratio, keep_cls=True)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, mid_states = self.transformer(x, attn_mask=x_mask, video_frame=video_frame)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if len(mid_states['attns']) == 0:
            assert mask_ratio > 0., "Must pass the semantic layer~"

        return x, mae_mask, mae_ids_restore, mid_states