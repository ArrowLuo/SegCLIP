# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------

import argparse
import os
import os.path as osp

import mmcv
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import set_random_seed
from mmseg.apis import single_gpu_test, multi_gpu_test
from omegaconf import OmegaConf, read_write
from seg_segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
from seg_segmentation.config import get_config
from seg_segmentation.logger import get_logger
from seg_segmentation.datasets import COCOObjectDataset, PascalContextDataset, PascalVOCDataset
from modules.tokenization_clip import SimpleTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import SegCLIP

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

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
    parser = argparse.ArgumentParser('SegCLIP seg_segmentation evaluation')
    parser.add_argument('--cfg', type=str, default="seg_segmentation/default.yml", help='path to config file',)
    parser.add_argument('--opts', help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+',)

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', type=str, help='root of output folder, the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument('--local_rank', type=int, required=True, help='local rank for DistributedDataParallel')

    parser.add_argument('--dataset', default='coco', choices=['voc', 'coco', 'context'], help='dataset classes')

    parser.add_argument("--pretrained_clip_name", type=str, default="ViT-B/16", help="Name to eval", )

    parser.add_argument('--max_words', type=int, default=77, help='')
    parser.add_argument('--max_frames', type=int, default=1, help='')

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--first_stage_layer', type=int, default=10, help="First stage layer.")

    args = parser.parse_args()

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

@torch.no_grad()
def validate_seg(cfg, config, data_loader, model, is_dist=True):
    logger = get_logger()
    if is_dist:
        dist.barrier()
    model.eval()

    if hasattr(model, 'module'):
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    text_transform = Tokenize(SimpleTokenizer(), max_seq_len=config.max_words)
    seg_model = build_seg_inference(model_without_ddp, data_loader.dataset, text_transform, cfg.evaluate.seg)

    if is_dist:
        mmddp_model = MMDistributedDataParallel(seg_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        mmddp_model.eval()
        results = multi_gpu_test(
            model=mmddp_model,
            data_loader=data_loader,
            tmpdir=None,
            gpu_collect=True,
            efficient_test=False,
            pre_eval=True,
            format_only=False)
    else:
        mmdp_model = MMDataParallel(seg_model, device_ids=[torch.cuda.current_device()])
        mmdp_model.eval()
        results = single_gpu_test(
            model=mmdp_model,
            data_loader=data_loader,
            pre_eval=True,
        )

    if dist.get_rank() == 0:
        metric = [data_loader.dataset.evaluate(results, metric='mIoU', logger='silent' if is_dist is False else None)]
    else:
        metric = [None]
    dist.broadcast_object_list(metric, src=0)
    miou_result = metric[0]['mIoU'] * 100

    torch.cuda.empty_cache()
    logger.info(f'Eval Seg mIoU {miou_result:.2f}')
    if is_dist:
        dist.barrier()
    return miou_result

def inference(args, cfg, model=None, is_dist=True):
    logger = get_logger()
    data_loader = build_seg_dataloader(build_seg_dataset(cfg.evaluate.seg), is_dist=is_dist)
    dataset = data_loader.dataset
    logger.info(f'Evaluating dataset: {dataset}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", cfg.local_rank)
    if model is None:
        model = init_model(args, device, 1, cfg.local_rank)

    if cfg.train.amp_opt_level != 'O0':
        model = amp.initialize(model, None, opt_level=cfg.train.amp_opt_level)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    miou= -1
    if 'seg' in cfg.evaluate.task:
        miou = validate_seg(cfg, args, data_loader, model, is_dist=is_dist)
        logger.info(f'mIoU of the network on the {len(data_loader.dataset)} test images: {miou:.2f}%')
    else:
        logger.info('No seg_segmentation evaluation specified')

    return miou

def eval_each_epoch(model, dataset='voc'):
    """
    Used as an inference in training process.
    :param model:
    :param dataset:
    :return:
    """
    def parse_args_each_epoch():
        args = argparse.Namespace()
        args.cfg = "seg_segmentation/default.yml"
        args.opts = None
        args.resume = False
        args.output = None
        args.tag = False
        args.local_rank = 0
        args.dataset = 'coco'
        args.pretrained_clip_name = "ViT-B/16"
        args.max_words = 77
        args.max_frames = 1
        args.init_model = None
        args.cache_dir = ""
        args.seed = 42
        args.first_stage_layer = 10
        cwd = os.path.dirname(os.path.abspath(__file__))
        args.cfg = os.path.join(cwd, args.cfg)
        return args

    args = parse_args_each_epoch()
    args.dataset = dataset
    cfg = get_config(args)

    if cfg.train.amp_opt_level != 'O0':
        assert amp is not None, 'amp not installed!'

    with read_write(cfg):
        cfg.evaluate.eval_only = True

    # =====================================================
    # Add for dataset
    # =====================================================
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
    # =====================================================
    torch.cuda.set_device(cfg.local_rank)

    set_random_seed(args.seed, use_rank_shift=True)
    cudnn.benchmark = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)

    if dist.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.json')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config saved to {path}')

    miou = inference(args, cfg, model, is_dist=False)
    return miou

def main():
    args = parse_args()
    cfg = get_config(args)

    if cfg.train.amp_opt_level != 'O0':
        assert amp is not None, 'amp not installed!'

    with read_write(cfg):
        cfg.evaluate.eval_only = True

    # =====================================================
    # Add for dataset
    # =====================================================
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
    # =====================================================

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f'RANK and WORLD_SIZE in environ: {rank}/{world_size}')
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(cfg.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()

    set_random_seed(args.seed, use_rank_shift=True)
    cudnn.benchmark = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)
    from util import logger_initialized
    logger_initialized['seg'] = logger

    if dist.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.json')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config saved to {path}')

    # print config
    logger.info(OmegaConf.to_yaml(cfg))

    inference(args, cfg)
    dist.barrier()


if __name__ == '__main__':
    main()
