from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import SegCLIP
from modules.optimization_adamw import AdaptAdamW

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT
import torch.cuda.amp as amp

torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='SegCLIP on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_vis", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/images_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=128, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=77, help='')
    parser.add_argument('--max_frames', type=int, default=1, help='')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.15, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--datatype", default="cc,coco,", type=str, help="Point the dataset to pretrain.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--lower_lr', type=float, default=0., help='lower lr for bert branch.')
    parser.add_argument('--lower_text_lr', type=float, default=0., help='lower lr for bert text branch.')

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--freeze_text_layer_num', type=int, default=0, help="Layer NO. of CLIP Text Encoder need to freeze.")
    parser.add_argument("--pretrained_clip_name", default="ViT-B/16", type=str, help="Choose a CLIP version")

    parser.add_argument('--use_vision_mae_recon', action='store_true', help="Use vision's mae to reconstruct the masked input image.")
    parser.add_argument('--use_text_mae_recon', action='store_true', help="Use text's mae to reconstruct the masked input text.")

    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight for optimizer.")
    parser.add_argument("--opt_b1", default=0.9, type=float, help="b1 for optimizer.")
    parser.add_argument("--opt_b2", default=0.98, type=float, help="b2 for optimizer.")
    parser.add_argument('--eps', default=1e-6, type=float)
    parser.add_argument('--lr_start', default=0., type=float, help='initial warmup lr (Note: rate for `--lr`)')
    parser.add_argument('--lr_end', default=0., type=float, help='minimum final lr (Note: rate for `--lr`)')
    parser.add_argument('--use_pin_memory', action='store_true', help="Use pin_memory when load dataset.")
    parser.add_argument('--clip_grad', default=1., type=float, help='value of clip grad.')

    parser.add_argument('--first_stage_layer', type=int, default=10, help="First stage layer.")

    parser.add_argument("--mae_vis_mask_ratio", default=0.75, type=float, help="mae vis mask ratio.")
    parser.add_argument("--mae_seq_mask_ratio", default=0.15, type=float, help="mae seq mask ratio.")

    parser.add_argument('--use_seglabel', action='store_true', help="Use Segmentation Label for Unsupervised Learning.")

    parser.add_argument('--disable_amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')

    args = parser.parse_args()
    args.disable_amp = True

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_pretrain and not args.do_train and not args.do_eval and not args.do_vis:
        raise ValueError("At least one of `do_pretrain`, `do_train`, `do_eval`, or `do_vis` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

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

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    prefix_ = "clip."
    clip_params = [(n, p) for n, p in param_optimizer if prefix_ in n]
    other_params = [(n, p) for n, p in param_optimizer if prefix_ not in n]

    clip_params_freeze = []
    clip_text_params_freeze = []
    clip_params_train = []
    for n, p in clip_params:
        if n.find("clip.visual.class_embedding") == 0 \
                or n.find("clip.visual.positional_embedding") == 0 \
                or n.find("clip.visual.conv1.") == 0 or n.find("clip.visual.ln_pre.") == 0 \
                or n.find("clip.logit_scale") == 0 or n.find("clip.ln_final.") == 0 \
                or n.find("clip.text_projection") == 0:
            clip_params_freeze.append((n, p))
            continue  # need to train0

        elif n.find("clip.positional_embedding") == 0 or n.find("clip.token_embedding.") == 0:
            clip_text_params_freeze.append((n, p))
            continue  # need to train0
        elif n.find("clip.visual.transformer.layers0.") == 0:  # make all image layer freeze
            clip_params_freeze.append((n, p))
            continue  # need to train
        elif n.find("clip.transformer.resblocks.") == 0:  # make all text layer freeze
            clip_params_freeze.append((n, p))
            continue  # need to train

        clip_params_train.append((n, p))
        if args.local_rank == 0:
            logger.info("Larger Lr: {}.".format(n))

    clip_params_freeze_decay = [p for n, p in clip_params_freeze if not any(nd in n for nd in no_decay)]
    clip_params_freeze_no_decay = [p for n, p in clip_params_freeze if any(nd in n for nd in no_decay)]
    clip_text_params_freeze_decay = [p for n, p in clip_text_params_freeze if not any(nd in n for nd in no_decay)]
    clip_text_params_freeze_no_decay = [p for n, p in clip_text_params_freeze if any(nd in n for nd in no_decay)]
    clip_params_train_decay = [p for n, p in clip_params_train if not any(nd in n for nd in no_decay)]
    clip_params_train_no_decay = [p for n, p in clip_params_train if any(nd in n for nd in no_decay)]
    other_params_decay = [p for n, p in other_params if not any(nd in n for nd in no_decay)]
    other_params_no_decay = [p for n, p in other_params if any(nd in n for nd in no_decay)]

    weight_decay = args.weight_decay
    eps = args.eps
    lower_lr = args.lower_lr
    if lower_lr == 0.:
        lower_lr = args.lr * coef_lr

    lower_text_lr = args.lower_text_lr
    if args.lower_text_lr == 0.:
        lower_text_lr = lower_lr

    optimizer_grouped_parameters = [
        {'params': clip_params_freeze_decay, 'weight_decay': weight_decay, 'lr': lower_lr},
        {'params': clip_params_freeze_no_decay, 'weight_decay': 0.0, 'lr': lower_lr},
        {'params': clip_text_params_freeze_decay, 'weight_decay': weight_decay, 'lr': lower_text_lr},
        {'params': clip_text_params_freeze_no_decay, 'weight_decay': 0.0, 'lr': lower_text_lr},
        {'params': clip_params_train_decay, 'weight_decay': weight_decay, 'lr': args.lr},
        {'params': clip_params_train_no_decay, 'weight_decay': 0.0, 'lr': args.lr},
        {'params': other_params_decay, 'weight_decay': weight_decay},
        {'params': other_params_no_decay, 'weight_decay': 0.0}
    ]

    scheduler = None
    opt_b1, opt_b2 = args.opt_b1, args.opt_b2
    weight_decay = args.weight_decay
    optimizer = AdaptAdamW(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=opt_b1, b2=opt_b2, e=eps,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0, lr_start=args.lr_start, lr_end=args.lr_end)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    scaler = amp.GradScaler(enabled=not args.disable_amp)

    return optimizer, scheduler, model, scaler

def save_model(epoch, args, model, optimizer, tr_loss, scaler, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            'scaler': scaler.state_dict(),
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = SegCLIP.from_pretrained(cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, scaler, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        image_seg = None
        if len(batch) == 6:
            input_ids, input_mask, segment_ids, image, coord, image_seg = batch
        else:
            input_ids, input_mask, segment_ids, image, coord = batch

        with amp.autocast(enabled=not args.disable_amp):
            loss = model(input_ids, segment_ids, input_mask, image, image_seg=image_seg)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

        if not args.disable_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += float(loss) if int(torch.isnan(loss)) == 0 else 0.
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            # Skip the loss with NAN manually.
            if int(torch.isnan(loss)) == 1:
                if local_rank == 0: logger.info("Note: loss is NAN (maybe caused by some wrong inputs).")
            else:
                if not args.disable_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Scaler:%.1f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss), scaler.get_scale(),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def eval_epoch(args, model, device, n_gpu):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
    model.eval()
    from main_seg_zeroshot import eval_each_epoch
    with torch.no_grad():
        miou = eval_each_epoch(model)
    return miou

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    model = init_model(args, device, n_gpu, args.local_rank)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args is None or args.local_rank == 0:
        logger.info("Number of params: {}".format(n_parameters))

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            FIRST_STAGE_LAYER = args.first_stage_layer
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue  # need to train
            elif name.find("visual.transformer.layers0.") == 0:  # make all image layer freeze
                layer_num = int(name.split(".layers0.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue  # need to train
            elif name.find("visual.transformer.layers2.") == 0:  # make all image layer freeze
                layer_num = int(name.split(".layers2.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num-FIRST_STAGE_LAYER:
                    continue  # need to train
            elif name.find("transformer.resblocks.") == 0:   # make all text layer freeze
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue  # need to train
            elif name.find("visual.transformer.semantic_layer1") == 0:
                continue  # need to train
            elif name.find("visual.transformer.semantic_layer2") == 0:
                continue  # need to train
            elif name.find("visual.transformer.layers_mae") == 0:
                continue  # need to train
            elif name.find("visual.transformer.reconstruct_layer") == 0:
                continue  # need to train

            # paramenters which < freeze_layer_num will be freezed
            param.requires_grad = False
            if args.local_rank == 0:
                logger.info("Freeze: {}.".format(name))

    if hasattr(model, "clip") and args.freeze_text_layer_num > 0:
        for name, param in model.clip.named_parameters():
            if name.find("positional_embedding") == 0 or name.find("token_embedding.weight") == 0:
                param.requires_grad = False
                if args.local_rank == 0:
                    logger.info("Freeze: {}.".format(name))
            elif name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num < args.freeze_text_layer_num:         # make text layer which less than `args.freeze_text_layer_num` freeze
                    param.requires_grad = False
                    if args.local_rank == 0:
                        logger.info("Freeze: {}.".format(name))

    if hasattr(model, "clip") and args.pretrained_clip_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14"]:
        for name, param in model.clip.named_parameters():
            if name.find("visual.positional_embedding") == 0 or name.find("visual.conv1.weight") == 0:
                param.requires_grad = False
                if args.local_rank == 0:
                    logger.info("Freeze: {}.".format(name))

    ## ####################################
    # dataloader loading
    ## ####################################
    if args.do_pretrain is False:
        assert args.datatype in DATALOADER_DICT, "If there are multiple dataset with `,`, the args.do_pretrain must be True."
    else:
        if args.local_rank == 0:
            logger.info("Pretrain NOW!!!!!!!!!")

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train or args.do_pretrain:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model, scaler = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch'] + 1
            # resumed_loss = checkpoint['loss']
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()

        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, scaler, local_rank=args.local_rank)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, scaler, type_name="")
                logger.info("Eval on val dataset")
                miou = eval_epoch(args, model, device, n_gpu)
                logger.info("The model has saved in: {}, the mIoU is: {:.2f}%".format(output_model_file, miou))

    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, device, n_gpu)

    elif args.do_vis:
        raise NotImplementedError()

if __name__ == "__main__":
    main()
