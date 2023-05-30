import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F

class RotationalSGD(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        weight_decay=required,
        nesterov=False,
        *,
        scale_invariance='channel',
        scale_invariance_min_dim=2,
        zero_mean=True,
        update_norm_decay_factor=0.99,
        eps=1e-8,
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
        if scale_invariance not in ['channel', 'tensor', False]:
            raise ValueError(f"Invalid {scale_invariance=}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            scale_invariance=scale_invariance,
            scale_invariance_min_dim=scale_invariance_min_dim,
            zero_mean=zero_mean,
            update_norm_decay_factor=update_norm_decay_factor,
            eps=eps,
            rotational=rotational,
        )
        super().__init__(params, defaults)

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
            eps = group['eps']

            # rotational Arguments
            scale_invariance = group['scale_invariance']
            scale_invariance_min_dim = group['scale_invariance_min_dim']
            undf = group['update_norm_decay_factor']

            for p in group['params']:
                param_state = self.state[p]

                if p.grad is None:
                    continue
                d_p = p.grad

                # rotational update for scale invariante parameters
                if 'rotational' in group and group['rotational'] is not None:
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

                if not rotational and weight_decay != 0.0:
                    # Only applied to scale sensitive parameters
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
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
                    # rotational update for scale invariant weights
                    # The other optimizers will more or less have the same modification here
                    if 'weight_norm' not in param_state:
                        if group['zero_mean']:
                            p_zero = zero_mean(p, scale_invariance)
                            param_state['weight_norm'] = tensor_norm(p_zero, scale_invariance).detach()
                        else:
                            param_state['weight_norm'] = tensor_norm(p, scale_invariance).detach()

                    # Project here (could also do after scaling but should be similar overall)
                    dot_p = dot_product(d_p, p, scale_invariance)
                    ratio = (dot_p / param_state['weight_norm']**2)
                    d_p = d_p - p * ratio

                    d_p_norm2 = (tensor_norm(d_p, scale_invariance)**2).detach()
                    param_state['update_norm2'] = (1 - undf) * d_p_norm2 + undf * param_state.get('update_norm2',0)
                    param_state['step'] = param_state.get('step', 0) + 1

                    avg_update_norm = torch.sqrt(param_state['update_norm2'] / (1 - undf**param_state['step']))
                    eta_r = (2*group['lr']*group['weight_decay']/(1+group['momentum']))**0.5
                    p_new = p - eta_r * (d_p / (avg_update_norm + eps)) * param_state['weight_norm']
                    if group['zero_mean']:
                        p_new = zero_mean(p_new, scale_invariance)
                    p_new = p_new * param_state['weight_norm'] / tensor_norm(p_new, scale_invariance)
                    p.copy_(p_new)
                else:
                    # Traditional update for other parameters
                    p.add_(d_p, alpha=-group['lr'])

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
