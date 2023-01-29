"""
Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""
from collections import OrderedDict
from typing import Tuple, Union

import hashlib
import os
import urllib
import warnings
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from modules.module_clip_util import LayerNorm, QuickGELU

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask=None):
        if attn_mask is not None:
            if hasattr(attn_mask, '__call__'):
                attn_mask_ = attn_mask(x.size()[0])   # LND
            else:
                ext_attn_mask = attn_mask.unsqueeze(1)
                ext_attn_mask = (1.0 - ext_attn_mask) * -1000000.0
                ext_attn_mask = ext_attn_mask.expand(-1, attn_mask.size(1), -1)
                attn_mask_ = ext_attn_mask.repeat_interleave(self.n_head, dim=0)
        else:
            attn_mask_ = None

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x_tuple:tuple):
        x, attn_mask, video_frame = x_tuple
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask, video_frame)

class TextTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask=None, video_frame=-1):
        return self.resblocks((x, attn_mask, video_frame))[0]