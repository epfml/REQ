""" AdamW Optimizer
Impl copied from PyTorch master
"""
import math
import torch
from torch.optim.optimizer import Optimizer

from shared.optimizers.common import center_rotational_weights, perform_rotational_update

class RotationalAdamW(Optimizer):
    r"""Implements a Rotational Variant of the AdamW algorithm.
    === Original AdamW Documentation ===
    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`.
    The AdamW variant, incorporating weight decay directly, was proposed in `Decoupled Weight 
    Decay Regularization`.

    Arguments:
        params (iterable): An iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients used for computing
            running averages of the gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): Term added to the denominator to improve
            numerical stability (default: 1e-8).
        weight_decay (float, optional): Weight decay coefficient (default: 1e-2).

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    === Additional Rotational Information ===
    This Rotational Variant of the original optimizer controls the average
    rotation (angular updates) of neurons or layers, aiming to match
    rotational equilibrium. It also constrains the norms of the weights to be fixed. 
    If fixed norms are not desired, learnable gains can be added to your model (similar to 
    e.g., shared.modules.normalization.SLinear) or via Weight Standardization (e.g., 
    shared.modules.normalization.WSLinear). Parameters not treated as rotational are 
    updated according to the original optimizer's scheme. By default, all neuronal weight 
    vectors (i.e., each convolutional filter or matrix row corresponding to a single output 
    feature) are treated as rotational. Single-dimensional vectors, such as biases and gains, 
    are by default treated as non-rotational, i.e., they are updated via the original
    optimizer. For additional details, see: https://arxiv.org/abs/2305.17212

    Arguments:
        update_norm_decay_factor (float, optional): Used to compute a running
            exponential average of the update magnitude, controlling the average angle. 
            Lower values result in stricter control over angular updates, while higher 
            values allow more variance between batches. Zero enforces strict adherence to
            the expected equilibrium rotation. Intuitively, larger batch sizes can tolerate
            lower values (default: 0.9).
        per_neuron (bool, optional): When True, the rotation and weight norm of each neuron 
            or output feature (i.e., rows of matrices, convolutional filters) are controlled 
            individually. If False, this control is applied at the tensor level, corresponding 
            to entire layers (default: True).
        zero_mean (bool, optional): When True, the weights of each neuron/tensor (according to 
            per_neuron) are centered, i.e., the mean component of the weights is removed at 
            initialization and from every update. This emulates the dynamics in Weight 
            Standardization, also known as Centered Weight Normalization (default: True).
        rotational_eps (float, optional): Added to the denominator when dividing by the 
            exponential moving average of the update size to prevent division by zero.
        rotational (bool, optional): Sets the default behavior for rotational updates in the 
            parameter group. Can be used to disable rotational updates for certain parameter
            groups. By default, rotational updates are applied to all parameter tensors with a 
            dimension of at least 2 that have weight decay applied (>0).
    """

    def __init__(
            # Standard Arguments
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            # Rotational Arguments
            update_norm_decay_factor=0.9,
            per_neuron=True,
            zero_mean=True,
            rotational_eps=1e-8,
            rotational=None,
    ):
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
            # Rotational Arguments
            update_norm_decay_factor=update_norm_decay_factor,
            per_neuron=per_neuron,
            zero_mean=zero_mean,
            rotational_eps=rotational_eps,
            rotational=rotational,
        )
        super().__init__(params, defaults)

        if zero_mean:
            center_rotational_weights(self.param_groups)

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

                if (rotational := group['rotational']) is None:
                    rotational = group['weight_decay'] != 0 and p.dim() > 1

                if rotational:
                    # By default apply rotational updates to matrices and higher dimensional
                    # tensors that weight decay is applied to
                    assert group['weight_decay'] != 0  # This would result in zero updates
                    avg_rotation = (2*group['lr']*weight_decay*(1-beta1)/(1+beta1))**0.5
                    d_p = torch.div(exp_avg, denom) / bias_correction1
                    perform_rotational_update(p, d_p, state, group, avg_rotation)
                else:
                    # Standard update with weight decay
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    step_size = group['lr'] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

