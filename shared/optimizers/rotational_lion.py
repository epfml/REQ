""" Sea Lion Optimizer
Paper: `Symbolic Discovery of Optimization Algorithms` - https://arxiv.org/abs/2302.06675
Original Impl: https://github.com/google/automl/tree/master/lion
"""

import torch
from torch.optim.optimizer import Optimizer
import math

from shared.optimizers.common import center_rotational_weights, perform_rotational_update

class RotationalLion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        # Rotational Arguments
        update_norm_decay_factor=0.9,
        per_neuron=True,
        zero_mean=True,
        rotational_eps=1e-8,
        rotational=None,
    ):
        """ Implements a Rotational Variant of the Lion optimizer.
        === Original Lion Documentation ===
        Initialize the hyperparameters.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-4)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.99))
            weight_decay (float, optional): weight decay coefficient (default: 0)

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
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                d_p = torch.sign(state['exp_avg'].mul(beta1).add_(p.grad, alpha=1 - beta1))
                state['exp_avg'].lerp_(p.grad, 1 - beta2)

                if (rotational := group['rotational']) is None:
                    rotational = group['weight_decay'] != 0 and p.dim() > 1

                if rotational:
                    # By default apply rotational updates to matrices and higher dimensional
                    # tensors that weight decay is applied to
                    assert group['weight_decay'] != 0  # This would result in zero updates
                    avg_rotation = (math.pi/2.0)**0.5 * (2*lr*weight_decay)**0.5 \
                        * ((1-beta1)**2 + beta1**2*(1-beta2)/(1+beta2))**0.5
                    perform_rotational_update(p, d_p, state, group, avg_rotation)
                else:
                    # Standard update with weight decay
                    p.mul_(1 - lr * weight_decay)
                    p.add_(d_p, alpha=-lr)

        return loss
