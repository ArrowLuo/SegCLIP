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
import argparse
import os
import os.path as osp
import sys

parentdir = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, parentdir)

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose
from omegaconf import read_write
from seg_segmentation.datasets import COCOObjectDataset, PascalContextDataset, PascalVOCDataset
from seg_segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_demo_pipeline, build_seg_inference
from seg_segmentation.config import get_config
from seg_segmentation.logger import get_logger

from modules.tokenization_clip import SimpleTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import SegCLIP


class Tokenize:

    def __init__(self, tokenizer, max_seq_len=77, truncate=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncate = truncate

    def __call__(self, texts):
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True

        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), self.max_seq_len, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.max_seq_len:
                if self.truncate:
                    tokens = tokens[:self.max_seq_len]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f'Input {texts[i]} is too long for context length {self.max_seq_len}')
            result[i, :len(tokens)] = torch.tensor(tokens)

        if expanded_dim:
            return result[0]

        return result


def parse_args():
    parser = argparse.ArgumentParser('SegCLIP demo')
    parser.add_argument('--cfg', type=str, default="seg_segmentation/default.yml", help='path to config file',)
    parser.add_argument('--opts', help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+',)

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--vis', help='Specify the visualization mode, could be a list, support "input", "pred", '\
                                      '"input_pred", "all_groups", "first_group", "final_group", "input_pred_label"', default=None, nargs='+')

    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--dataset', default='voc', choices=['voc', 'coco', 'context'], help='dataset classes for visualization')

    parser.add_argument('--input', type=str, help='input image path')
    parser.add_argument('--output_dir', type=str, default="output", help='output dir')

    parser.add_argument("--pretrained_clip_name", type=str, default="ViT-B/16", help="Name to eval", )

    parser.add_argument('--max_words', type=int, default=77, help='')
    parser.add_argument('--max_frames', type=int, default=1, help='')

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--first_stage_layer', type=int, default=10, help="First stage layer.")

    args = parser.parse_args()
    args.local_rank = 0  # compatible with config

    cwd = os.path.dirname(os.path.abspath(__file__))
    args.cfg = os.path.join(cwd, args.cfg)

    return args

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = SegCLIP.from_pretrained(cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model

def inference(args, cfg):
    device = torch.device(args.device)
    model = init_model(args, device, 1, cfg.local_rank)
    model = revert_sync_batchnorm(model)
    model.eval()

    text_transform = Tokenize(SimpleTokenizer(), max_seq_len=args.max_words)

    if args.dataset == 'voc':
        dataset_class = PascalVOCDataset
        seg_cfg = 'seg_segmentation/configs/_base_/datasets/pascal_voc12.py'
    elif args.dataset == 'coco':
        dataset_class = COCOObjectDataset
        seg_cfg = 'seg_segmentation/configs/_base_/datasets/coco.py'
    elif args.dataset == 'context':
        dataset_class = PascalContextDataset
        seg_cfg = 'seg_segmentation/configs/_base_/datasets/pascal_context.py'
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    with read_write(cfg):
        cwd = os.path.dirname(os.path.abspath(__file__))
        seg_cfg = os.path.join(cwd, seg_cfg)
        cfg.evaluate.seg.cfg = seg_cfg
        if args.input in ["coco", "voc", 'context']:
            cfg.evaluate.seg.opts = ['test_cfg.mode=whole']
        else:
            cfg.evaluate.seg.opts = ['test_cfg.mode=slide']

    seg_model = build_seg_inference(model, dataset_class, text_transform, cfg.evaluate.seg)

    if args.input in ["coco", "voc", 'context']:
        cfg_ss = mmcv.Config.fromfile(cfg.evaluate.seg.cfg)
        print(cfg_ss.data.test)

        data_loader = build_seg_dataloader(build_seg_dataset(cfg.evaluate.seg))
        # dataset = data_loader.dataset

        seg_num_ = 0
        loader_indices = data_loader.batch_sampler
        for batch_indices, data in zip(loader_indices, data_loader):
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for batch_idx, img, img_meta in zip(batch_indices, imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]
                img_show = img_show[:224, :224, :]
                img_show = mmcv.imresize(img_show, (224, 224))
                vis_seg(seg_model, img_show, args.output_dir, args.vis, img_idx=str(batch_idx))
                seg_num_ += 1
            if seg_num_ > 10:
                break
    else:
        input_ = args.input
        input_ = mmcv.imread(input_)
        vis_seg(seg_model, input_, args.output_dir, args.vis, img_idx=os.path.splitext(os.path.basename(args.input))[0])


def vis_seg(seg_model, input_img, output_dir, vis_modes, img_idx=None):
    device = next(seg_model.parameters()).device
    test_pipeline = build_seg_demo_pipeline(img_size=224)
    # prepare data
    data = dict(img=input_img)

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if next(seg_model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    with torch.no_grad():
        result = seg_model(return_loss=False, rescale=True, **data)

    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)

    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        for vis_mode in vis_modes:
            if img_idx is not None:
                out_file = osp.join(output_dir, 'vis_imgs', vis_mode, f'{vis_mode}_{img_idx}.jpg')
            else:
                out_file = osp.join(output_dir, 'vis_imgs', vis_mode, f'{vis_mode}.jpg')

            seg_model.show_result(img_show, img_tensor.to(device), result, out_file, vis_mode)


def main():
    args = parse_args()
    cfg = get_config(args)

    with read_write(cfg):
        cfg.evaluate.eval_only = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)
    from util import logger_initialized
    logger_initialized['seg'] = logger

    inference(args, cfg)


if __name__ == '__main__':
    main()
