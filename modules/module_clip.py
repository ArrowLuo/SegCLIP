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

from modules.module_clip_util import _MODELS
from modules.module_clip_util import random_masking, convert_weights
from modules.module_clip_util import LayerNorm, CLIP_Module
from modules.module_clip_vtransformer import VisualTransformer
from modules.module_clip_ttransformer import TextTransformer
from modules.module_clip_util import ProjectMLP

class CLIP(CLIP_Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # vision linear of patch
                 first_stage_layer: int = 10,
                 ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisualTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width,
                                        layers=vision_layers, heads=vision_heads, output_dim=embed_dim,
                                        first_stage_layer=first_stage_layer,
        )


        self.transformer = TextTransformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))

        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # self.logit_scale = nn.Parameter(torch.ones([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    @property
    def dtype(self):
        if hasattr(self.visual, "proj") and isinstance(self.visual.proj, torch.Tensor):
            return self.visual.proj.dtype
        elif hasattr(self.visual, "conv1") and hasattr(self.visual.conv1, "weight"):
            return self.visual.conv1.weight.dtype
        elif hasattr(self.visual, "class_embedding"):
            return self.visual.class_embedding.dtype
        elif hasattr(self.transformer, "class_embedding"):
            return self.transformer.class_embedding.dtype

        return torch.float32

    def encode_image_hidden_ln(self, image, video_frame=-1, mask_ratio=0.):
        """
        Refactor this function due to we need hidden_ln
        """
        hidden, mae_mask, mae_ids_restore, mid_states = self.visual(image.type(self.dtype), video_frame=video_frame, mask_ratio=mask_ratio)
        hidden_ln = self.visual.ln_post(hidden)
        return hidden_ln, mae_mask, mae_ids_restore, mid_states

    def encode_image(self, image, return_hidden=False, video_frame=-1, mask_ratio=0.):
        hidden_ln, mae_mask, mae_ids_restore, mid_states = self.encode_image_hidden_ln(image, video_frame=video_frame, mask_ratio=mask_ratio)
        if isinstance(self.visual.proj, torch.Tensor):
            hidden = hidden_ln @ self.visual.proj
        else:
            hidden = self.visual.proj(hidden_ln)

        x = hidden[:, 0, :]

        if mask_ratio > 0.: assert return_hidden is True
        if return_hidden:
            if mask_ratio > 0.:
                return x, hidden, mae_mask, mae_ids_restore, mid_states
            return x, hidden, mid_states
        return x

    def encode_text(self, text, attn_mask=None, return_hidden=False, mask_ratio=0.):
        if attn_mask is None:
            attn_mask = self.build_attention_mask

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x + pos_emd

        # MAE ===>
        mae_mask, mae_ids_restore = None, None
        if mask_ratio > 0.:
            assert return_hidden is True
            # masking: length -> length * mask_ratio
            x, mae_mask, mae_ids_restore, ids_keep = random_masking(x, mask_ratio, keep_cls=True, keep_sep=True,
                                                                    sep_pos=text.argmax(dim=-1),)
            if hasattr(attn_mask, '__call__') is False:
                attn_mask = torch.gather(attn_mask, dim=1, index=ids_keep)
            text = torch.gather(text, dim=1, index=ids_keep)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        hidden_ln = self.ln_final(x).type(self.dtype)

        if isinstance(self.text_projection, torch.Tensor):
            hidden = hidden_ln @ self.text_projection
        else:
            hidden = self.text_projection(hidden_ln)

        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]

        if return_hidden:
            if mask_ratio > 0.:
                return x, hidden, mae_mask, mae_ids_restore
            return x, hidden

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, attn_mask=self.build_attention_mask)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            if hasattr(block, "attn"):   # compatible with GAU
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            if hasattr(block, "mlp"):   # compatible with GAU
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None and isinstance(self.text_projection, torch.Tensor):
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())
