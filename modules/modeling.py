from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from modules.util_module import dist_collect, show_log, update_attr, check_attr, get_attr
from modules.util_module import PreTrainedModel, AllGather, CrossEn

from modules.module_clip import CLIP, available_models
from modules.module_mae import MAEDecoder
from util import get_logger

allgather = AllGather.apply

class SegCLIPPreTrainedModel(PreTrainedModel, nn.Module):
    def __init__(self, *inputs, **kwargs):
        super(SegCLIPPreTrainedModel, self).__init__()
        self.clip = None

    @classmethod
    def from_pretrained(cls, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = get_attr(task_config, "pretrained_clip_name", default_value="ViT-B/16", donot_log=True)

        if pretrained_clip_name in available_models():
            clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        else:
            # We will reset ViT but keep Text Encoder
            clip_state_dict = CLIP.get_config(pretrained_clip_name="ViT-B/32")

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        for key, val in clip_state_dict.items():
            # HARD CODE for initialization trick
            FIRST_STAGE_LAYER = 10
            if hasattr(task_config, "first_stage_layer"):
                FIRST_STAGE_LAYER = task_config.first_stage_layer

            new_key = "clip." + key
            if "visual.transformer." in key:
                _, _, _, _, n_, *_ = new_key.split(".")
                n_ = int(n_)
                if n_ >= FIRST_STAGE_LAYER:
                    new_key = new_key.replace(".resblocks.", ".layers2.")
                    new_key_ls_ = new_key.split(".")
                    new_key_ls_[4] = str(n_ - FIRST_STAGE_LAYER)
                    new_key = ".".join(new_key_ls_)
                else:
                    new_key = new_key.replace(".resblocks.", ".layers0.")
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        model = cls(clip_state_dict, *inputs, **kwargs)

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config, print_logger=get_logger())

        return model


class SegCLIP(SegCLIPPreTrainedModel):
    def __init__(self, clip_state_dict, task_config):
        super(SegCLIP, self).__init__()
        self.task_config = task_config
        self.ignore_image_index = -1

        pretrained_clip_name = get_attr(task_config, "pretrained_clip_name", default_value="ViT-B/16", donot_log=True)
        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len([k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size

            if pretrained_clip_name not in available_models():
                assert pretrained_clip_name[:5] == "ViT-B"
                vision_patch_size = int(pretrained_clip_name.split("/")[-1])
                assert image_resolution % vision_patch_size == 0
                grid_size = image_resolution // vision_patch_size
                show_log(task_config, "\t\t USE {} NOW!!!!!!!!!!!!".format(pretrained_clip_name))
        else:
            raise NotImplementedError()

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.first_stage_layer = get_attr(task_config, "first_stage_layer", default_value=10)

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            first_stage_layer=self.first_stage_layer,
        ).float()
        self.clip = nn.SyncBatchNorm.convert_sync_batchnorm(self.clip)
        # <=== End of CLIP Encoders

        self.loss_fct = CrossEn()
        self.loss_fct_stdce = nn.CrossEntropyLoss()

        ## ==============================================================================
        # Reconstruct the masked input as MAE
        ## ==============================================================================
        mae_vis_mask_ratio = get_attr(task_config, "mae_vis_mask_ratio", default_value=0.75)
        self.use_vision_mae_recon = get_attr(task_config, "use_vision_mae_recon", default_value=False)
        if self.use_vision_mae_recon:
            self.vis_mask_ratio = mae_vis_mask_ratio
            decoder_embed_dim = vision_width // 2
            decoder_num_heads = 8
            vision_patch_size_ = vision_patch_size
            self.vis_mae_decoder = MAEDecoder(vision_width, decoder_embed_dim, image_resolution,
                                              vision_patch_size_,
                                              decoder_depth=3, decoder_num_heads=decoder_num_heads, mlp_ratio=4.,
                                              norm_layer=partial(nn.LayerNorm, eps=1e-6))

        mae_seq_mask_ratio = get_attr(task_config, "mae_seq_mask_ratio", default_value=0.15)
        self.use_text_mae_recon = get_attr(task_config, "use_text_mae_recon", default_value=False)
        if self.use_text_mae_recon:
            self.seq_mask_ratio = mae_seq_mask_ratio
            decoder_embed_dim = embed_dim // 2
            decoder_num_heads = 8
            vision_patch_size_ = vision_patch_size
            self.seq_mae_decoder = MAEDecoder(embed_dim, decoder_embed_dim, image_resolution,
                                              vision_patch_size_,
                                              decoder_depth=3, decoder_num_heads=decoder_num_heads, mlp_ratio=4.,
                                              choice_seq=True,
                                              pred_len=vocab_size, seq_len=self.task_config.max_words)

        ## ==============================================================================
        # Use segmentation label for unsupervised learning
        ## ==============================================================================
        self.use_seglabel = get_attr(task_config, "use_seglabel", default_value=False)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, image, image_seg=None):

        # B x T x L
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        # T x 3 x H x W
        image_input = torch.as_tensor(image).float()
        b, pair, channel, h, w = image_input.shape
        image = image_input[:, 0].view(b, channel, h, w)
        image_frame = 1   # TODO: HARD CODE, A compatibility for video in CLIP4Clip

        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         image, shaped=True,
                                                                         image_frame=image_frame, return_hidden=True)

        if isinstance(sequence_output, tuple):
            sequence_output, sequence_hidden = sequence_output
        if isinstance(visual_output, tuple):
            visual_output, visual_hidden, mid_states = visual_output

        if self.use_seglabel:
            # T x patch_len x patch_len
            image_seg_input = torch.as_tensor(image_seg)
            image_seg = image_seg_input[:, 0]

        if self.training:
            loss = 0.

            sim_matrix_t2v, sim_matrix_v2t = self._loose_similarity(sequence_output, visual_output)
            labels = torch.arange(sequence_output.size(0), dtype=torch.long, device=sequence_output.device)
            labels = labels + sequence_output.size(0) * self.task_config.rank
            sim_loss1 = self.loss_fct_stdce(sim_matrix_t2v, labels)
            sim_loss2 = self.loss_fct_stdce(sim_matrix_v2t, labels)
            sim_loss = (sim_loss1 + sim_loss2) / 2.
            loss = loss + sim_loss

            if self.use_seglabel:
                mid_attn_hidden = mid_states['attns'][0]['hard_attn'].permute(0, 2, 1) # B x L x CENTER
                image_seg_ = image_seg.view(b, -1)
                image_seg_ = image_seg_.unsqueeze(-1) - image_seg_.unsqueeze(-2)
                image_seg_ = (image_seg_ == 0).to(dtype=mid_attn_hidden.dtype)     # B x L x L
                clutering_sum = torch.einsum('b g l, b l c -> b g c', image_seg_, mid_attn_hidden)
                clutering_mean = clutering_sum / torch.clamp_min(torch.sum(image_seg_, dim=-1, keepdim=True), min=1.0)

                coef_ = mid_attn_hidden.size(0) * mid_attn_hidden.size(1) * mid_attn_hidden.size(2)
                kl_mean_1 = F.kl_div(F.log_softmax(mid_attn_hidden, dim=-1), F.softmax(clutering_mean, dim=-1), reduction='sum') / float(coef_)
                kl_mean_2 = F.kl_div(F.log_softmax(clutering_mean, dim=-1), F.softmax(mid_attn_hidden, dim=-1), reduction='sum') / float(coef_)
                clutering_loss = (kl_mean_1 + kl_mean_2) / 2.
                loss = loss + clutering_loss

            if self.use_text_mae_recon:
                sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True,
                                                           return_hidden=True, mask_ratio=self.seq_mask_ratio)
                _, seq_hidden, seq_mae_mask, seq_mae_ids_restore = sequence_output

                seq_mae_mask = seq_mae_mask.view(-1, seq_mae_mask.size(-1))
                seq_mae_ids_restore = seq_mae_ids_restore.view(-1, seq_mae_ids_restore.size(-1))

                _mae_mask = (seq_mae_mask + attention_mask).gt(1)
                seq_mae_loss = self.seq_mae_decoder.forward_seq(input_ids, seq_hidden, _mae_mask, seq_mae_ids_restore, attention_mask)
                loss = loss + seq_mae_loss

            if self.use_vision_mae_recon:
                visual_output = self.get_visual_output(image, shaped=True, image_frame=image_frame,
                                                       return_hidden=True, mask_ratio=self.vis_mask_ratio)
                _, vis_hidden, vis_mae_mask, vis_mae_ids_restore, mid_mae_states = visual_output

                vis_hidden = mid_mae_states['hidden']
                cls_ = torch.mean(vis_hidden, dim=1, keepdim=True)
                vis_hidden = torch.cat([cls_, vis_hidden], dim=1)

                vis_mae_mask = vis_mae_mask.view(-1, vis_mae_mask.size(-1))
                vis_mae_ids_restore = vis_mae_ids_restore.view(-1, vis_mae_ids_restore.size(-1))

                vis_mae_loss = self.vis_mae_decoder.forward_vis(image, vis_hidden, vis_mae_mask, vis_mae_ids_restore,
                                                                loss_allpatch=False)
                loss = loss + vis_mae_loss

            return loss
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False, return_hidden=False, seq_model=None, mask_ratio=0.):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        if seq_model is None:
            seq_model = self.clip

        bs_pair = input_ids.size(0)
        sequence_hidden = seq_model.encode_text(input_ids, return_hidden=return_hidden, mask_ratio=mask_ratio)

        if isinstance(sequence_hidden, tuple):
            if mask_ratio > 0:
                sequence_hidden = tuple([itm.float().view(bs_pair, -1, itm.size(-1)) for itm in sequence_hidden[:2]]
                                        + [itm.view(bs_pair, -1, itm.size(-1)) for itm in sequence_hidden[2:]])
            else:
                sequence_hidden = tuple([itm.float().view(bs_pair, -1, itm.size(-1)) for itm in sequence_hidden])
        else:
            sequence_hidden = sequence_hidden.float().view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, image, shaped=False, image_frame=-1, return_hidden=False, vis_model=None, mask_ratio=0.):
        if shaped is False:
            image_input = torch.as_tensor(image).float()
            b, pair, channel, h, w = image_input.shape
            image = image_input[:, 0].view(b, channel, h, w)
            image_frame = 1   # TODO: HARD CODE, A compatibility for video in CLIP4Clip

        if vis_model is None:
            vis_model = self.clip

        bs_pair = image.size(0)
        visual_hidden = vis_model.encode_image(image, video_frame=image_frame, return_hidden=return_hidden, mask_ratio=mask_ratio)
        if isinstance(visual_hidden, tuple):
            if mask_ratio > 0:
                visual_hidden = tuple([itm.float().view(bs_pair, -1, itm.size(-1)) for itm in visual_hidden[:2]]
                                      + [itm.view(bs_pair, -1, itm.size(-1)) for itm in visual_hidden[2:4]] + [visual_hidden[4]])
            else:
                visual_hidden = tuple([itm.float().view(bs_pair, -1, itm.size(-1)) for itm in visual_hidden[:2]]
                                      + [visual_hidden[2]])
        else:
            visual_hidden = visual_hidden.float().view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, image,
                                   shaped=False, image_frame=-1, return_hidden=False, seq_model=None, vis_model=None):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

            image_input = torch.as_tensor(image).float()
            b, pair, channel, h, w = image_input.shape
            image = image_input[:, 0].view(b, channel, h, w)
            image_frame = 1   # TODO: HARD CODE, A compatibility for video in CLIP4Clip

        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True, return_hidden=return_hidden, seq_model=seq_model)
        visual_output = self.get_visual_output(image, shaped=True, image_frame=image_frame, return_hidden=return_hidden, vis_model=vis_model)

        return sequence_output, visual_output

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output,):
        image_out = torch.mean(visual_output, dim=1)
        return image_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        image_out = self._mean_pooling_for_similarity_visual(visual_output)
        return text_out, image_out

    def _loose_similarity(self, sequence_output, visual_output, logit_scale=None):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        visual_output = visual_output.squeeze(1)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        if logit_scale is not None:
            logit_scale = torch.clamp(logit_scale.exp(), max=100)
        else:
            logit_scale = torch.clamp(self.clip.logit_scale.exp(), max=100)
        if self.training:
            visual_output_collect = dist_collect(visual_output, self.task_config)
            sequence_output_collect = dist_collect(sequence_output, self.task_config)
            torch.distributed.barrier()

            retrieve_logits_t2v = logit_scale * torch.matmul(sequence_output, visual_output_collect.t())
            retrieve_logits_v2t = logit_scale * torch.matmul(visual_output, sequence_output_collect.t())
        else:
            retrieve_logits_t2v = logit_scale * torch.matmul(sequence_output, visual_output.t())
            retrieve_logits_v2t = retrieve_logits_t2v.T

        return retrieve_logits_t2v, retrieve_logits_v2t


    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, shaped=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        contrastive_direction = ()
        retrieve_logits_t2v, retrieve_logits_v2t = self._loose_similarity(sequence_output, visual_output)

        return retrieve_logits_t2v, retrieve_logits_v2t, contrastive_direction
