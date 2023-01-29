# Adapted from: https://github.com/facebookresearch/mae/blob/main/models_mae.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Adapted from: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Adapted from: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
from torch import nn
import numpy as np

def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * 3))
    return x

def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * patch_size, h * patch_size))
    return imgs

# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/4f4a192f0fd272102c8852b00b1007dffd292b90/transformer/Models.py#L11
def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return position_enc

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
# ====================================================================

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.n_head = n_head
        self.norm1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(d_model)
        self.mlp = Mlp(in_features=d_model, hidden_features=int(d_model * mlp_ratio), act_layer=act_layer, drop=drop)

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
        x, attn_mask = x_tuple
        x = x + self.drop_path(self.attention(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return (x, attn_mask)

class MAEDecoder(nn.Module):

    def __init__(self, embed_dim, decoder_embed_dim, image_resolution, patch_size,
                 decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, in_chans=3, choice_seq=False, pred_len=None, seq_len=None):

        super().__init__()

        if choice_seq:
            assert pred_len is not None
            assert seq_len is not None
        else:
            pred_len = patch_size ** 2 * in_chans

        self.choice_seq = choice_seq
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.patch_size = patch_size

        self.num_patches = (image_resolution // patch_size) ** 2
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1 if choice_seq is False else seq_len,
                                                          decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        if choice_seq is False:
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])
        else:
            self.decoder_blocks = nn.Sequential(*[
                ResidualAttentionBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, norm_layer=norm_layer)
                for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, pred_len, bias=True)  # decoder to patch

        self.alm_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.choice_seq:
            decoder_pos_embed = position_encoding_init(self.seq_len, self.decoder_pos_embed.shape[-1])
        else:
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches ** .5),
                                                        cls_token=True)

        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_vis(self, image, vis_hidden, vis_mae_mask, vis_mae_ids_restore, loss_allpatch=False):
        # embed tokens
        vis_hidden = self.decoder_embed(vis_hidden)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(vis_hidden.shape[0], vis_mae_ids_restore.shape[1] - vis_hidden.shape[1], 1)
        x_ = torch.cat([vis_hidden, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=vis_mae_ids_restore.unsqueeze(-1).repeat(1, 1, vis_hidden.shape[2]))  # unshuffle
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        pred = x[:, 1:, :]

        target = patchify(image, patch_size=self.patch_size)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        if loss_allpatch:
            loss = loss.mean()
        else:
            loss = (loss * vis_mae_mask[:, 1:]).sum() / vis_mae_mask[:, 1:].sum()  # mean loss on removed patches

        return loss

    def forward_seq(self, input_ids, seq_hidden, seq_mae_mask, seq_mae_ids_restore, attention_mask):
        # embed tokens
        seq_hidden = self.decoder_embed(seq_hidden)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(seq_hidden.shape[0], seq_mae_ids_restore.shape[1] - seq_hidden.shape[1], 1)
        x_ = torch.cat([seq_hidden, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=seq_mae_ids_restore.unsqueeze(-1).repeat(1, 1, seq_hidden.shape[2]))  # unshuffle
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.decoder_blocks((x, attention_mask))[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.decoder_norm(x)
        # predictor projection
        pred = self.decoder_pred(x)
        # remove cls token

        input_ids = input_ids.view(-1) * seq_mae_mask.view(-1).to(dtype=input_ids.dtype) - \
                    (1-seq_mae_mask.view(-1).to(dtype=input_ids.dtype))
        loss = self.alm_loss_fct(pred.view(-1, self.pred_len), input_ids)

        return loss