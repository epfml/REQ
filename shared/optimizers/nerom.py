# Originally based on:
# https://github.com/jxbz/nero/blob/32436c0305e1428611497f6dcefd45809e5b08d2/nero.py
# This version adds momentum
# mom_beta: momentum coefficient (i.e. beta1 for Adam)
# nesterov: True/False or float representing current grad proportion
# rms_beta: rms decay coefficient (i.e. beta2 for Adam)
# update_normalization: rms (Adam like) or sign (Lion like)

import torch
from torch.optim.optimizer import Optimizer

def neuron_norm(x):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        return x.norm(dim=1).view(*view_shape)
    else:
        return x.abs()


def neuron_mean(x):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        return x.mean(dim=1).view(*view_shape)
    else:
        raise Exception("neuron_mean not defined on 1D tensors.")


class NeroM(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        mom_beta=0.99,
        nesterov=False,
        rms_beta=0.999,
        update_normalization='rms',
        sigma_b=0.01,
        constraints=True,
    ):
        self.beta = rms_beta
        self.constraints = constraints
        self.sigma_b = sigma_b
        defaults = dict(
            lr=lr, mom_beta=mom_beta, nesterov=nesterov,
            rms_beta=rms_beta, sigma_b=sigma_b, update_normalization=update_normalization,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                if self.constraints and p.dim() > 1:
                    p.data -= neuron_mean(p)
                    p.data /= neuron_norm(p)
                state = self.state[p]
                state['step'] = 0
                state['velocity'] = torch.zeros_like(p)
                if group['update_normalization'] == 'rms':
                    state['exp_avg_sq'] = torch.zeros_like(neuron_norm(p))
                state['scale'] = neuron_norm(p).mean()
                if state['scale'] == 0.0:
                    # FIXME: This could be brittle, e.g. when fine tuning etc
                    # NOTE: Initial scale for gammas is just their value (will break for zero init gain)
                    state['scale'] = group['sigma_b']

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                # Probably no need to bother with bias correction for momentum
                state['velocity'] = group['mom_beta']*state['velocity'] + (1-group['mom_beta']) * p.grad

                if group['nesterov']:
                    factor = 1 - group['mom_beta'] if group['nesterov'] == True else group['nesterov']
                    update = (1-factor) * state['velocity'] + factor*p.grad
                else:
                    update = state['velocity']

                state['step'] += 1
                if group['update_normalization'] == 'rms':
                    bias_correction = 1 - group['rms_beta'] ** state['step']
                    state['exp_avg_sq'] = group['rms_beta'] * state['exp_avg_sq'] + (1-group['rms_beta']) * neuron_norm(p.grad)**2
                    update = update / (state['exp_avg_sq']/bias_correction).sqrt()
                    update[torch.isnan(update)] = 0
                elif group['update_normalization'] == 'sign':
                    update = torch.sign(update)
                else:
                    assert not group['update_normalization'], f"{group['update_normalization']=}"
                
                p.data -= group['lr'] * state['scale'] * update

                if self.constraints and p.dim() > 1:
                    p.data -= neuron_mean(p)
                    p.data /= neuron_norm(p)

        return loss
