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

_base_ = ['../custom_import.py']
# dataset settings
dataset_type = 'COCOObjectDataset'
data_root = '/home/root/datasets/coco_stuff164k'
img_norm_cfg = dict(mean=[122.7709383, 116.7460125, 104.09373615], std=[68.5005327, 66.6321579, 70.32316305], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 224),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val2017',
        ann_dir='annotations/val2017',
        pipeline=test_pipeline))

test_cfg = dict(bg_thresh=.65, mode='slide', stride=(224, 224), crop_size=(224, 224))
