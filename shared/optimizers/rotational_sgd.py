import torch
from torch.optim.optimizer import Optimizer, required

from shared.optimizers.common import center_rotational_weights, perform_rotational_update


class RotationalSGD(Optimizer):
    r"""Implements a Rotational Variant of the SGDM optimizer.
    === Original SGDM Documentation ===
    Implements stochastic gradient descent (optionally with momentum).

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

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
        self,
        params,
        lr=required,
        momentum=0,
        weight_decay=required,
        nesterov=False,
        # Rotational Arguments
        update_norm_decay_factor=0.9,
        per_neuron=True,
        zero_mean=True,
        rotational_eps=1e-8,
        rotational=None,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0):
            raise ValueError("Nesterov momentum requires a momentum")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
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
            momentum = group['momentum']
            nesterov = group['nesterov']

            for p in group['params']:
                if (rotational := group['rotational']) is None:
                    rotational = group['weight_decay'] != 0 and p.dim() > 1

                state = self.state[p]

                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0 and not rotational:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                if rotational:
                    assert group['weight_decay'] != 0  # This would result in zero updates
                    avg_rotation = (2*group['lr']*group['weight_decay']/(1+group['momentum']))**0.5
                    perform_rotational_update(p, d_p, state, group, avg_rotation)
                else:
                    # Standard update with weight decay
                    p.add_(d_p, alpha=-group['lr'])

        return loss

