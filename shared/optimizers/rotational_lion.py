""" Sea Lion Optimizer
Paper: `Symbolic Discovery of Optimization Algorithms` - https://arxiv.org/abs/2302.06675
Original Impl: https://github.com/google/automl/tree/master/lion
"""
# Copyright 2023 Google Research. All Rights Reserved.
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
# ==============================================================================
from typing import List

import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
import math


class RotationalLion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(
            self,
            params,
            lr=1e-4,
            betas=(0.9, 0.99),
            weight_decay=0.0,
            scale_invariance='channel',
            scale_invariance_min_dim=2,
            zero_mean=True,
            update_norm_decay_factor=0.99,
            is_linearized=False,
            eps=1e-8,
            rotational=None,
    ):
        """Initialize the hyperparameters.

        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            scale_invariance=scale_invariance,
            scale_invariance_min_dim=scale_invariance_min_dim,
            zero_mean=zero_mean,
            update_norm_decay_factor=update_norm_decay_factor,
            is_linearized=is_linearized,
            eps=eps,
            rotational=None)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.

        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            weight_norms = []
            update_norms = []
            beta1, beta2 = group['betas']
            is_zero_mean = group['zero_mean']
            scale_invariant = group['scale_invariance']
            scale_invariance = group['scale_invariance']
            scale_invariance_min_dim = group['scale_invariance_min_dim']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if scale_invariant:
                        state['update_norm2'] =  torch.tensor([0.0], device=p.device)
                        if is_zero_mean:
                            param_zero = zero_mean(p, scale_invariance)
                            state['weight_norm'] = tensor_norm(param_zero, scale_invariance).detach()
                        else:
                            state['weight_norm'] = tensor_norm(p, scale_invariance).detach()

                state['step'] += 1
                exp_avgs.append(state['exp_avg'])
                weight_norms.append(state['weight_norm'])
                update_norms.append(state['update_norm2'])

                # rotational update for scale invariante parameters
                if 'rotational' in group and group['rotational'] is not None:
                    rotational = group['rotational']
                else:
                    rotational = torch.squeeze(p).dim() >= scale_invariance_min_dim and scale_invariance and weight_decay > 0

            lion(
                params_with_grad,
                grads,
                exp_avgs,
                update_norm2s=update_norms,
                weight_norms=weight_norms,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=weight_decay,
                scale_invariance=scale_invariant,
                scale_invariance_min_dim=scale_invariance_min_dim,
                is_zero_mean=is_zero_mean,
                update_norm_decay_factor=group['update_norm_decay_factor'],
                is_linearized=group['is_linearized'],
                step=state['step'],
                eps=group['eps'],
                rotational=rotational)

        return loss


def lion(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        update_norm2s: List[torch.Tensor],
        weight_norms: List[torch.Tensor],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        scale_invariance: bool,
        scale_invariance_min_dim: int,
        is_zero_mean=bool,
        update_norm_decay_factor=float,
        is_linearized: bool,
        step: int,
        eps: float,
        rotational: bool,
):
    
    r"""Functional API that performs Lion algorithm computation.
    """
    _single_tensor_lion(
        params,
        grads,
        exp_avgs,
        update_norm2s=update_norm2s,
        weight_norms=weight_norms,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        scale_invariance=scale_invariance,
        scale_invariance_min_dim=scale_invariance_min_dim,
        is_zero_mean=is_zero_mean,
        undf=update_norm_decay_factor,
        is_linearized=is_linearized,
        step=step,
        eps=eps,
        rotational=rotational,
    )


def _single_tensor_lion(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        update_norm2s: List[torch.Tensor],
        weight_norms: List[torch.Tensor],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        scale_invariance: bool,
        scale_invariance_min_dim: int,
        is_zero_mean:bool,
        is_linearized: bool,
        undf:float,
        step: int,
        eps: float,
        rotational: bool):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        update_norm2 = update_norm2s[i]
        weight_norm = weight_norms[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            param = torch.view_as_real(param)

        # Weight update
        update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)

        if is_linearized:
            update = update / torch.sqrt(torch.mean(update**2))
        else:
            update = torch.sign(update)

        if rotational:
            # Check if we can apply rotational updates here
            try:
                assert not (torch.squeeze(param).dim() == 1 and scale_invariance == 'channel')
                assert weight_decay > 0  # Otherwise the rotation will be zero
            except Exception as e:
                raise ValueError('rotational update is applied in an invalid scenario.')

            update = update - param * (dot_product(update, param, scale_invariance) / weight_norm**2)
            gradient_update_norm2 = (tensor_norm(update, scale_invariance)**2).detach()
            update_norm2.data = (1 - undf) * gradient_update_norm2 + undf * update_norm2
 
            # param project here (could also do after scaling but should be similar overall)
            avg_update_norm = torch.sqrt(update_norm2 / (1 - undf**step))

            etar = (math.pi/2.0)**0.5 * (2*lr*weight_decay)**0.5 * ((1-beta1)**2 + beta1**2*(1-beta2)/(1+beta2))**0.5
            param_new = param - etar * (update / (eps + avg_update_norm)) * weight_norm
            if is_zero_mean:
               param_new = zero_mean(param_new, scale_invariance)
            param_new = param_new * weight_norm / tensor_norm(param_new, scale_invariance)
            param.copy_(param_new)
        else:
            #perform stepweight decay
            param.mul_(1 - lr * weight_decay)
            param.add_(update, alpha=-lr)

        # Decay the momentum running average coefficient
        exp_avg.lerp_(grad, 1 - beta2)

def tensor_norm(tensor, scale_invariance):
    if scale_invariance == 'tensor':
        return torch.linalg.vector_norm(tensor)
    elif scale_invariance == 'channel':
        # This assumes weights are stored as K x other dims
        # Which is the default for both Linear and Conv2d
        norm = torch.linalg.vector_norm(tensor.view(tensor.shape[0], -1), dim=1)
        return norm.view(-1, *([1]*(tensor.dim()-1)))
    else:
        raise ValueError(f"Invalid {scale_invariance=}")


def dot_product(a, b, scale_invariance):
    if scale_invariance == 'tensor':
        return torch.sum(a*b)
    elif scale_invariance == 'channel':
        # This assumes weights are stored as K x other dims
        # Which is the default for both Linear and Conv2d
        return (a.flatten(1)*b.flatten(1)).sum(dim=1).view(a.shape[0], *([1]*(a.dim()-1)))
    else:
        raise ValueError(f"Invalid {scale_invariance=}")


def zero_mean(tensor, scale_invariance):
    if scale_invariance == 'tensor':
        return tensor - tensor.mean()
    elif scale_invariance == 'channel':
        # This assumes weights are stored as K x other dims
        # Which is the default for both Linear and Conv2d
        flat_tensor = tensor.view(tensor.shape[0], -1)
        return (flat_tensor - flat_tensor.mean(dim=1, keepdim=True)).view_as(tensor)
    else:
        raise ValueError(f"Invalid {scale_invariance=}")
