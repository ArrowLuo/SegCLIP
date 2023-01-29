# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
import logging

logger = logging.getLogger(__name__)

def warmup_cosine(x, warmup=0.002, lr_start=0., lr_end=0.):
    if x < warmup:
        return (x * (1. - lr_start) / warmup) + lr_start
    new_x = (x-warmup) / (1-warmup)
    return lr_end + 0.5 * (1. - lr_end) * (1 + math.cos(math.pi * new_x))

def warmup_constant(x, warmup=0.002, lr_start=0., lr_end=0.):
    """ Linearly increases learning rate over `warmup`*`t_total` (as provided to AdaptAdamW) training steps.
        Learning rate is 1. afterwards. """
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002, lr_start=0., lr_end=0.):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to AdaptAdamW) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.), 0)

SCHEDULES = {
    'warmup_cosine':   warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear':   warmup_linear,
}


class AdaptAdamW(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0, lr_start=0., lr_end=0.):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        if not 0.0 <= lr_start < 1.0:
            raise ValueError("Invalid lr_start parameter: {} - should be in [0.0, 1.0[".format(lr_start))
        if not 0.0 <= lr_end < 1.0:
            raise ValueError("Invalid lr_end parameter: {} - should be in [0.0, 1.0[".format(lr_end))

        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm, lr_start=lr_start, lr_end=lr_end)
        super(AdaptAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'],
                                                              group['lr_start'], group['lr_end'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['b1'], group['b2']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                # exp_avg.mul_(beta1).add_(1 - beta1, grad) --> pytorch 1.7
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad) --> pytorch 1.7
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['e'])

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    progress = state['step']/group['t_total']
                    lr_scheduled = group['lr'] * schedule_fct(progress, group['warmup'],
                                                              group['lr_start'], group['lr_end'])
                else:
                    lr_scheduled = group['lr']

                # TODO: Perform stepweight decay, move to here, which is different from official adamw.py
                p.mul_(1 - lr_scheduled * group['weight_decay'])

                step_size = lr_scheduled / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
