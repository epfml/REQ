import math
import torch
from torch.optim import Optimizer
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

from shared.optimizers.common import dot_product, zero_mean, tensor_norm, center_rotational_weights

class RotationalWrapper(Optimizer):
    def __init__(
        self,
        params,
        inner_type='adamw',
        etar_func=None,
        update_norm_decay_factor=0.9,
        per_neuron=True,
        zero_mean=True,
        rotational_eps=1e-8,
        **inner_hyperparameters,
    ):
        # Allow choosing etar func via string
        if isinstance(etar_func, str):
            if etar_func == 'adamw':
                self.etar_func = adamw_etar_func
            elif etar_func == 'sgdm':
                self.etar_func = sgdm_etar_func
            elif etar_func == 'lion':
                self.etar_func = lion_etar_func
            else:
                raise ValueError(f"Unknown {etar_func=}")
        else:
            self.etar_func = etar_func

        if inner_type == 'adamw':
            self.init_state_get_lists = adamw_init_state_get_lists
            self.get_inner_update = adamw_get_update

            if self.etar_func is None:
                self.etar_func = adamw_etar_func

            # Set defaults based on official PyTorch implementation
            inner_hyperparameters = {
                'lr': 1e-3,
                'weight_decay': 1e-2,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                **inner_hyperparameters,
            }

        if inner_type == 'adam':
            self.init_state_get_lists = adam_init_state_get_lists
            self.get_inner_update = adam_get_update

            if self.etar_func is None:
                # Adam does not have a regular rotational rate that we can
                # enforce throughout training (i.e. depends on the gradnorm)
                raise ValueError("No etar_func provided for Adam")

            # Set defaults based on official PyTorch implementation
            inner_hyperparameters = {
                'lr': 1e-3,
                'weight_decay': 0.0,  # Note that this will result in non-rotational updates
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                **inner_hyperparameters,
            }

        if inner_type == 'sgdm':
            self.init_state_get_lists = sgdm_init_state_get_lists
            self.get_inner_update = sgdm_get_update

            if self.etar_func is None:
                self.etar_func = sgdm_etar_func

            # Set defaults based on official PyTorch implementation
            inner_hyperparameters = {
                # lr required
                'momentum': 0.0,
                'weight_decay': 0.0,  # Note that this will result in non-rotational updates
                **inner_hyperparameters,
            }

        if inner_type == 'lion':
            self.init_state_get_lists = lion_init_state_get_lists
            self.get_inner_update = lion_get_update

            if self.etar_func is None:
                self.etar_func = lion_etar_func

            # Set defaults based on official PyTorch implementation
            inner_hyperparameters = {
                'lr': 1e-4,
                'weight_decay': 1.0,
                'betas': (0.9, 0.99),
                'eps': 1e-8,
                **inner_hyperparameters,
            }

        defaults = dict(
            per_neuron=per_neuron,
            update_norm_decay_factor=update_norm_decay_factor,
            zero_mean=zero_mean,
            rotational_eps=rotational_eps,
            **inner_hyperparameters,
        )
        super().__init__(params, defaults)

        # Change initialization to be zero mean
        if zero_mean:
            center_rotational_weights(self.param_groups)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            state_lists = self.init_state_get_lists(self.state, group)
            # These do not include the -lr factor:
            unscaled_delta_grads, unscaled_delta_lambdas = self.get_inner_update(state_lists, group)

            for p in group['params']:
                if group.get('rotational') is not None:
                    rotational = group['rotational']
                    if rotational:
                        assert group['weight_decay'] != 0  # This results in zero updates
                else:
                    # By default matrices and higher order tensors are treated
                    # as scale invariant if rotational is not explicitly set
                    # and weight decay is applied to them
                    rotational = group['weight_decay'] != 0 and p.dim() > 1

                if rotational:
                    state = self.state[p]

                    if 'norm' not in state:
                        # Save initial norm
                        if group['zero_mean']:
                            p_zero = zero_mean(p, group['per_neuron'])
                            state['norm'] = tensor_norm(p_zero, group['per_neuron'])
                        else:
                            state['norm'] = tensor_norm(p, group['per_neuron'])

                    # Project update to be orthogonal to p
                    update = unscaled_delta_grads[p]
                    projection = p * (dot_product(update, p, group['per_neuron']) / state['norm']**2)
                    update = update - projection

                    if group['zero_mean']:
                        update = zero_mean(update, group['per_neuron'])

                    # Keep track of the average update norm (exponential moving root-mean-square)
                    undf = group['update_norm_decay_factor']
                    square_update_norm = tensor_norm(update, group['per_neuron'])**2
                    state['update_norm_sq'] = (1-undf) * square_update_norm + undf * state.get('update_norm_sq', 0)

                    # Bias correction
                    avg_update_norm = torch.sqrt(state['update_norm_sq'] / (1 - undf**state['step']))

                    # Get rotational rate
                    eta_r = self.etar_func(group)

                    # Compute update
                    p_new = p - eta_r * state['norm'] * (update / (avg_update_norm + group['rotational_eps']))

                    # Project back onto the sphere
                    p_new = p_new * state['norm'] / tensor_norm(p_new, group['per_neuron'])

                    # Write update
                    p.copy_(p_new)
                else:
                    # Standard non-rotational update
                    p.add_(-group['lr']*(unscaled_delta_grads[p]+unscaled_delta_lambdas[p]))


################################################################################
# AdamW functions, based on PyTorch 2.0 source code

def adamw_init_state_get_lists(optimizer_state, group):
    state_lists = dict(
        params_with_grad=(params_with_grad := []),
        grads=(grads := []),
        exp_avgs=(exp_avgs := []),
        exp_avg_sqs=(exp_avg_sqs := []),
        state_steps=(state_steps := []),
    )

    for p in group["params"]:
        if p.grad is None:
            continue
        params_with_grad.append(p)
        if p.grad.is_sparse:
            raise RuntimeError("AdamW does not support sparse gradients")
        grads.append(p.grad)

        state = optimizer_state[p]

        # State initialization
        if len(state) == 0:
            state["step"] = torch.tensor(0.0)

            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )

            # Exponential moving average of squared gradient values
            state["exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )

        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])
        state_steps.append(state["step"])

    return state_lists


def adamw_get_update(state_lists, group):
    params = state_lists['params_with_grad']
    grads = state_lists['grads']
    exp_avgs = state_lists['exp_avgs']
    exp_avg_sqs = state_lists['exp_avg_sqs']
    state_steps = state_lists['state_steps']

    beta1, beta2 = group['betas']
    weight_decay = group['weight_decay']
    eps = group['eps']

    unscaled_delta_grads = dict()
    unscaled_delta_lambdas = dict()

    if len(params) == 0:
        return dict(), dict()

    grouped_tensors = _group_tensors_by_device_and_dtype([
        params, grads, exp_avgs, exp_avg_sqs, state_steps])
    for ((
        device_params,
        device_grads,
        device_exp_avgs,
        device_exp_avg_sqs,
        device_state_steps,
    ), _) in grouped_tensors.values():

        # update steps
        torch._foreach_add_(device_state_steps, 1)

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
        bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]

        bias_correction2_sqrt = [_dispatch_sqrt(bc) for bc in bias_correction2]

        exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
        torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
        torch._foreach_add_(exp_avg_sq_sqrt, eps)

        scaled_updates = torch._foreach_div(device_exp_avgs, exp_avg_sq_sqrt)
        torch._foreach_div_(scaled_updates, bias_correction1)
    
        unscaled_delta_wd_list = torch._foreach_mul(device_params, weight_decay)
        for param, udg, udl in zip(device_params, scaled_updates, unscaled_delta_wd_list):
            unscaled_delta_grads[param] = udg
            unscaled_delta_lambdas[param] = udl

    # Does not include -lr factor
    return unscaled_delta_grads, unscaled_delta_lambdas


def adamw_etar_func(group):
    # Computes the equilibrium eta_r
    lr = group['lr']
    wd = group['weight_decay']
    beta1 = group['betas'][0]
    return (2*lr*wd*(1-beta1)/(1+beta1))**0.5


# /AdamW functions
################################################################################
# Adam functions, based on PyTorch 2.0 source code

def adam_init_state_get_lists(optimizer_state, group):
    state_lists = dict(
        params_with_grad=(params_with_grad := []),
        grads=(grads := []),
        grad_exp_avgs=(grad_exp_avgs := []),
        l2_exp_avgs=(l2_exp_avgs := []),
        total_exp_avg_sqs=(total_exp_avg_sqs := []),
        state_steps=(state_steps := []),
    )

    for p in group["params"]:
        if p.grad is None:
            continue
        params_with_grad.append(p)
        if p.grad.is_sparse:
            raise RuntimeError("Adam does not support sparse gradients")
        grads.append(p.grad)

        state = optimizer_state[p]

        # State initialization
        if len(state) == 0:
            state["step"] = torch.tensor(0.0)

            # Exponential moving average of gradient values
            state["grad_exp_avg"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )

            # Exponential moving average of the l2 component
            state["l2_exp_avg"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )

            # Exponential moving average of the total square l2+grad values
            state["total_exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )

        grad_exp_avgs.append(state["grad_exp_avg"])
        l2_exp_avgs.append(state["l2_exp_avg"])
        total_exp_avg_sqs.append(state["total_exp_avg_sq"])
        state_steps.append(state["step"])

    return state_lists


def adam_get_update(state_lists, group):
    params = state_lists['params_with_grad']
    grads = state_lists['grads']
    grad_exp_avgs = state_lists['grad_exp_avgs']
    l2_exp_avgs = state_lists['l2_exp_avgs']
    total_exp_avg_sqs = state_lists['total_exp_avg_sqs']
    state_steps = state_lists['state_steps']

    beta1, beta2 = group['betas']
    weight_decay = group['weight_decay']
    eps = group['eps']

    unscaled_delta_grads = dict()
    unscaled_delta_lambdas = dict()

    if len(params) == 0:
        return dict(), dict()

    grouped_tensors = _group_tensors_by_device_and_dtype([
        params, grads, grad_exp_avgs, l2_exp_avgs, total_exp_avg_sqs, state_steps
    ])
    for ((
        device_params,
        device_grads,
        device_grad_exp_avgs,
        device_l2_exp_avgs,
        device_total_exp_avg_sqs,
        device_state_steps,
    # )) in grouped_tensors.values():
    ), _) in grouped_tensors.values():
        # update steps
        torch._foreach_add_(device_state_steps, 1)

        # Update first momentum (separately for the grad and l2 term)
        torch._foreach_lerp_(device_grad_exp_avgs, device_grads, 1 - beta1)
        l2_term = torch._foreach_mul(device_params, weight_decay)
        torch._foreach_lerp_(device_l2_exp_avgs, l2_term, 1 - beta1)
        del l2_term

        # Update the second momentum (after adding the l2 and grad together)
        device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)
        torch._foreach_mul_(device_total_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_total_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
        bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]

        bias_correction2_sqrt = [_dispatch_sqrt(bc) for bc in bias_correction2]

        total_exp_avg_sq_sqrt = torch._foreach_sqrt(device_total_exp_avg_sqs)
        torch._foreach_div_(total_exp_avg_sq_sqrt, bias_correction2_sqrt)
        torch._foreach_add_(total_exp_avg_sq_sqrt, eps)

        unscaled_grad_updates = torch._foreach_div(device_grad_exp_avgs, total_exp_avg_sq_sqrt)
        torch._foreach_div_(unscaled_grad_updates, bias_correction1)

        unscaled_l2_updates = torch._foreach_div(device_l2_exp_avgs, total_exp_avg_sq_sqrt)
        torch._foreach_div_(unscaled_l2_updates, bias_correction1)

        for param, udg, udl in zip(device_params, unscaled_grad_updates, unscaled_l2_updates):
            unscaled_delta_grads[param] = udg
            unscaled_delta_lambdas[param] = udl

    # Does not include -lr factor
    return unscaled_delta_grads, unscaled_delta_lambdas

# /Adam functions
################################################################################
# SGDM functions, based on PyTorch 2.0 source code

def sgdm_init_state_get_lists(optimizer_state, group):
    state_lists = dict(
        params_with_grad=(params_with_grad := []),
        grads=(grads := []),
        grad_exp_avgs=(grad_exp_avgs := []),
        l2_exp_avgs=(l2_exp_avgs := []),
        state_steps=(state_steps := []),
    )

    for p in group["params"]:
        if p.grad is None:
            continue
        params_with_grad.append(p)
        if p.grad.is_sparse:
            raise RuntimeError("SGDM does not support sparse gradients")
        grads.append(p.grad)

        state = optimizer_state[p]

        # State initialization
        if len(state) == 0:
            state["step"] = torch.tensor(0.0)

            # Exponential moving average of gradient values
            state["grad_exp_avg"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
            # Exponential moving average of the l2 component
            state["l2_exp_avg"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )

        grad_exp_avgs.append(state["grad_exp_avg"])
        l2_exp_avgs.append(state["l2_exp_avg"])
        state_steps.append(state["step"])

    return state_lists


def sgdm_get_update(state_lists, group):
    params = state_lists['params_with_grad']
    grads = state_lists['grads']
    grad_exp_avgs = state_lists['grad_exp_avgs']
    l2_exp_avgs = state_lists['l2_exp_avgs']
    state_steps = state_lists['state_steps']

    momentum = group['momentum']
    weight_decay = group['weight_decay']

    assert not group.get('nesterov'), "This implementation doesn't support it"

    unscaled_delta_grads = dict()
    unscaled_delta_lambdas = dict()

    if len(params) == 0:
        return dict(), dict()

    grouped_tensors = _group_tensors_by_device_and_dtype([
        params, grads, grad_exp_avgs, l2_exp_avgs, state_steps
    ])
    for ((
        device_params,
        device_grads,
        device_grad_exp_avgs,
        device_l2_exp_avgs,
        device_state_steps,
    # )) in grouped_tensors.values():
    ), _) in grouped_tensors.values():
        # update steps (used in the rotational wrapper)
        torch._foreach_add_(device_state_steps, 1)

        # Update gradient momentum tracker
        torch._foreach_mul_(device_grad_exp_avgs, momentum)
        torch._foreach_add_(device_grad_exp_avgs, device_grads)

        # Update l2 momentum tracker
        torch._foreach_mul_(device_l2_exp_avgs, momentum)
        torch._foreach_add_(device_l2_exp_avgs, device_params, alpha=weight_decay)

        for param, udg, udl in zip(device_params, device_grad_exp_avgs, device_l2_exp_avgs):
            unscaled_delta_grads[param] = udg
            unscaled_delta_lambdas[param] = udl

    # Does not include -lr factor
    return unscaled_delta_grads, unscaled_delta_lambdas


def sgdm_etar_func(group):
    # Computes the equilibrium eta_r
    lr = group['lr']
    wd = group['weight_decay']
    alpha = group['momentum']
    return (2*lr*wd/(1+alpha))**0.5


#/SGDM functions
################################################################################
# Lion functions, based on PyTorch 2.0 source code

def lion_init_state_get_lists(optimizer_state, group):
    state_lists = dict(
        params_with_grad=(params_with_grad := []),
        grads=(grads := []),
        exp_avgs=(exp_avgs := []),
        state_steps=(state_steps := []),
    )

    for p in group["params"]:
        if p.grad is None:
            continue
        params_with_grad.append(p)
        if p.grad.is_sparse:
            raise RuntimeError("AdamW does not support sparse gradients")
        grads.append(p.grad)

        state = optimizer_state[p]

        # State initialization
        if len(state) == 0:
            state["step"] = torch.tensor(0.0)

            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )

        exp_avgs.append(state["exp_avg"])
        state_steps.append(state["step"])

    return state_lists


def lion_get_update(state_lists, group):
    params = state_lists['params_with_grad']
    grads = state_lists['grads']
    exp_avgs = state_lists['exp_avgs']
    state_steps = state_lists['state_steps']

    beta1, beta2 = group['betas']
    weight_decay = group['weight_decay']
    eps = group['eps']

    unscaled_delta_grads = dict()
    unscaled_delta_lambdas = dict()

    if len(params) == 0:
        return dict(), dict()

    grouped_tensors = _group_tensors_by_device_and_dtype([
        params, grads, exp_avgs, state_steps])
    for ((
        device_params,
        device_grads,
        device_exp_avgs,
        device_state_steps,
    ), _) in grouped_tensors.values():
        # update steps (used in the rotational wrapper)
        torch._foreach_add_(device_state_steps, 1)

        if len(device_params) == 0:
            return

        device_grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads]
        device_exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avgs]
        device_params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_params]

        # Weight update
        updates = torch._foreach_mul(device_exp_avgs, beta1)
        torch._foreach_add_(updates, device_grads, alpha=1 - beta1)

        unscaled_updates = [u.sign() for u in updates]

        # Decay the momentum running average coefficient
        torch._foreach_mul_(device_exp_avgs, beta2)
        torch._foreach_add_(device_exp_avgs, device_grads, alpha=1 - beta2)
    
        unscaled_delta_wd_list = torch._foreach_mul(device_params, weight_decay)
        for param, udg, udl in zip(device_params, unscaled_updates, unscaled_delta_wd_list):
            unscaled_delta_grads[param] = udg
            unscaled_delta_lambdas[param] = udl

    # Does not include -lr factor
    return unscaled_delta_grads, unscaled_delta_lambdas

def lion_etar_func(group):
    # Computes the equilibrium eta_r
    lr = group['lr']
    wd = group['weight_decay']
    beta1 = group['betas'][0]
    beta2 = group['betas'][1]
    return (math.pi/2.0)**0.5 * (2*lr*wd)**0.5 * ((1-beta1)**2 + beta1**2*(1-beta2)/(1+beta2))**0.5


# /Lion functions
################################################################################
# Helper functions from torch.optim.optimizer

def _get_value(x):
    # item is significantly faster than a cpu tensor in eager mode
    if not torch.jit.is_scripting() and torch._utils.is_compiling():
        return x
    else:
        return x.item()

def _stack_if_compiling(x):
    if not torch.jit.is_scripting() and torch._utils.is_compiling()():
        return torch.stack(x)
    else:
        return x

def _dispatch_sqrt(x: float):  # float annotation is needed because of torchscript type inference
    if not torch.jit.is_scripting() and isinstance(x, torch.Tensor):
        return x.sqrt()
    else:
        return math.sqrt(x)
