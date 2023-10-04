""" AdamW Optimizer
Impl copied from PyTorch master

NOTE: Builtin optim.AdamW is used by the factory, this impl only serves as a Python based reference, will be removed
someday
"""
import math
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F


class RotationalSpeedControlledAdamW(Optimizer):
    r"""Implements rotational version of AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            scale_invariance='channel',
            scale_invariance_min_dim=2,
            zero_mean=True,
            update_norm_decay_factor=0.99,
            control_group_percentage=0.0,
            speed=100,
            is_double=False,
            rotational=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            scale_invariance=scale_invariance,
            scale_invariance_min_dim=scale_invariance_min_dim,
            zero_mean=zero_mean,
            update_norm_decay_factor=update_norm_decay_factor,
            rotational=rotational,
            control_group_percentage=control_group_percentage,
            is_double=is_double,
            speed=speed,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

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
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']

            # rotational Arguments
            scale_invariance = group['scale_invariance']
            scale_invariance_min_dim = group['scale_invariance_min_dim']
            undf = group['update_norm_decay_factor']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # rotational update for scale invariante parameters
                if group['rotational'] is not None:
                    rotational = group['rotational']
                else:
                    rotational = torch.squeeze(p).dim() >= scale_invariance_min_dim and scale_invariance and weight_decay > 0

                if rotational:
                    # Check if we can apply rotational updates here
                    try:
                        assert not (torch.squeeze(p).dim() == 1 and scale_invariance == 'channel')
                        assert weight_decay > 0  # Otherwise the rotation will be zero
                    except Exception as e:
                        raise ValueError('rotational update is applied in an invalid scenario.')

                    if 'weight_norm' not in state:
                        if group['zero_mean']:
                            p_zero = zero_mean(p, scale_invariance)
                            state['weight_norm'] = tensor_norm(p_zero, scale_invariance).detach()
                        else:
                            state['weight_norm'] = tensor_norm(p, scale_invariance).detach()

                    # compute velocity of adamW
                    d_p = torch.div(exp_avg, denom) 

                    # Project here (could also do after scaling but should be similar overall)
                    d_p = d_p - p * (dot_product(d_p, p, scale_invariance) / state['weight_norm']**2)
                    d_p_norm2 = (tensor_norm(d_p, scale_invariance)**2).detach()
                    state['update_norm2'] = (1 - undf) * d_p_norm2 + undf * state.get('update_norm2',0)

                    avg_update_norm = torch.sqrt(state['update_norm2'] / (1 - undf**state['step']))

                    step_size = (2*group['lr']*weight_decay*(1-beta1)/(1+beta1))**0.5

                    control_group_percentage = group['control_group_percentage']
                    if group['control_group_percentage'] > 0.0:
                        speed = group['speed']
                        if not group['is_double']:
                            control_group_size = int(control_group_percentage / 2 * d_p.shape[0])
                            d_p[:control_group_size, ...] *= 1 / speed
                            d_p[control_group_size:2*control_group_size, ...] *= speed
                        else:
                            control_group_size = int(control_group_percentage * d_p.shape[0])
                            d_p[:control_group_size, ...] *=  1 / speed

                    p_new = p - step_size * (d_p / (eps + avg_update_norm)) * state['weight_norm']
                    if group['zero_mean']:
                        p_new = zero_mean(p_new, scale_invariance)
                    p_new = p_new * state['weight_norm'] / tensor_norm(p_new, scale_invariance)
                    p.copy_(p_new)
                else:
                    # Perform stepweight decay
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    
                    step_size = group['lr'] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

def tensor_norm(tensor, scale_invariance):
    if scale_invariance == 'tensor':
        return torch.linalg.vector_norm(tensor)
    elif scale_invariance == 'channel':
        # This assumes weights are stored as K x other dims
        # Which is the default for both Linear and Conv2d
        norm = torch.linalg.vector_norm(tensor.reshape(tensor.shape[0], -1), dim=1)
        return norm.reshape(-1, *([1]*(tensor.dim()-1)))
    else:
        raise ValueError(f"Invalid {scale_invariance=}")


def dot_product(a, b, scale_invariance):
    if scale_invariance == 'tensor':
        return torch.sum(a*b)
    elif scale_invariance == 'channel':
        # This assumes weights are stored as K x other dims
        # Which is the default for both Linear and Conv2d
        return (a.flatten(1)*b.flatten(1)).sum(dim=1).reshape(a.shape[0], *([1]*(a.dim()-1)))
    else:
        raise ValueError(f"Invalid {scale_invariance=}")


def zero_mean(tensor, scale_invariance):
    if scale_invariance == 'tensor':
        return tensor - tensor.mean()
    elif scale_invariance == 'channel':
        # This assumes weights are stored as K x other dims
        # Which is the default for both Linear and Conv2d
        flat_tensor = tensor.reshape(tensor.shape[0], -1)
        return (flat_tensor - flat_tensor.mean(dim=1, keepdim=True)).view_as(tensor)
    else:
        raise ValueError(f"Invalid {scale_invariance=}")
