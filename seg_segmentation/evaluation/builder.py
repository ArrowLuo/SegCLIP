# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual
# property and proprietary rights in and to this software, related
# documentation and any modifications thereto.  Any use, reproduction,
# disclosure or distribution of this software and related documentation
# without an express license agreement from NVIDIA CORPORATION is strictly
# prohibited.
#
# Written by Jiarui Xu
# Adapted from https://github.com/NVlabs/GroupViT and Modified by Huaishao Luo
# -------------------------------------------------------------------------

import numpy as np
import mmcv
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.datasets.pipelines import Compose
from omegaconf import OmegaConf
from einops import rearrange, repeat
from ..misc import build_dataset_class_tokens

from .vit_seg import ViTSegInference


def build_seg_dataset(config):
    """Build a dataset from config."""
    cfg = mmcv.Config.fromfile(config.cfg)
    dataset = build_dataset(cfg.data.test)
    return dataset


def build_seg_dataloader(dataset, is_dist=True):

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=is_dist,
        shuffle=False,
        persistent_workers=True,
        pin_memory=False)
    return data_loader


def build_seg_inference(model, dataset, text_transform, config):
    cfg = mmcv.Config.fromfile(config.cfg)
    if len(config.opts):
        cfg.merge_from_dict(OmegaConf.to_container(OmegaConf.from_dotlist(OmegaConf.to_container(config.opts))))
    with_bg = dataset.CLASSES[0] == 'background'
    if with_bg:
        classnames = dataset.CLASSES[1:]
    else:
        classnames = dataset.CLASSES
    text_tokens = build_dataset_class_tokens(text_transform, config.template, classnames)   # [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH]

    text_tokens = text_tokens.to(next(model.parameters()).device)
    num_classes, num_templates = text_tokens.shape[:2]
    text_tokens = rearrange(text_tokens, 'n t l -> (n t) l', n=num_classes, t=num_templates)

    text_tokens = model.clip.encode_text(text_tokens)

    text_embedding = rearrange(text_tokens, '(n t) c -> n t c', n=num_classes, t=num_templates)   # [N, T, C]
    # text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding.mean(dim=1)
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    # text_embedding = model.build_text_embedding(text_embedding)

    kwargs = dict(with_bg=with_bg)
    if hasattr(cfg, 'test_cfg'):
        kwargs['test_cfg'] = cfg.test_cfg

    seg_model = ViTSegInference(model, text_embedding, **kwargs)

    seg_model.CLASSES = dataset.CLASSES
    seg_model.PALETTE = dataset.PALETTE

    return seg_model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
            img = mmcv.imread(results['img'])
        elif isinstance(results['img'], np.ndarray):
            results['filename'] = None
            results['ori_filename'] = None
            img = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
            img = mmcv.imread(results['img'])

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def build_seg_demo_pipeline(img_size=224,):
    """Build a demo pipeline from config."""
    img_norm_cfg = dict(mean=[122.7709383, 116.7460125, 104.09373615], std=[68.5005327, 66.6321579, 70.32316305], to_rgb=True)
    test_pipeline = Compose([
        LoadImage(),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, img_size),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ])
    return test_pipeline