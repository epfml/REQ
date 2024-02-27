import torch

def tensor_norm(tensor, per_neuron=True):
    if per_neuron:
        # This assumes weights are stored as K x other dims
        # Which is the default for both Linear and Conv2d
        norm = torch.linalg.vector_norm(tensor.reshape(tensor.shape[0], -1), dim=1)
        return norm.reshape(-1, *([1]*(tensor.dim()-1)))
    else:
        return torch.linalg.vector_norm(tensor)


def dot_product(a, b, per_neuron=True):
    if per_neuron:
        # This assumes weights are stored as K x other dims
        # Which is the default for both Linear and Conv2d
        return (a.flatten(1)*b.flatten(1)).sum(dim=1).reshape(a.shape[0], *([1]*(a.dim()-1)))
    else:
        return torch.sum(a*b)


def zero_mean(tensor, per_neuron=True):
    if per_neuron:
        # This assumes weights are stored as K x other dims
        # Which is the default for both Linear and Conv2d
        flat_tensor = tensor.reshape(tensor.shape[0], -1)
        return (flat_tensor - flat_tensor.mean(dim=1, keepdim=True)).view_as(tensor)
    else:
        return tensor - tensor.mean()


@torch.no_grad()
def center_rotational_weights(param_groups, verbose=True):
    if verbose:
        print("Centering Rotational Weight Vectors")

    for group in param_groups:
        for p in group['params']:
            if group.get('rotational') is not None:
                rotational = group['rotational']
            else:
                # By default matrices and higher order tensors are treated
                # as scale invariant if rotational is not explicitly set
                # and weight decay is applied to them
                rotational = group['weight_decay'] != 0 and p.dim() > 1

            if rotational:
                init_norm = tensor_norm(p, group['per_neuron'])
                p_zero = zero_mean(p, group['per_neuron'])
                p_zero = init_norm * p_zero / tensor_norm(p_zero, group['per_neuron'])
                p.copy_(p_zero)


@torch.no_grad()
def perform_rotational_update(p, d_p, state, group, avg_rotation):
    # p: The parameter tensor
    # d_p: Update component without any weight decay / L2 regularization and
    #   not scaled by the learning rate
    # state: The optimizer state dictionary for this parameter
    # group: The optimizer parameter group defining rotational hyperparameters etc
    # avg_rotation: The desired average rotation (varies between optimizers)
    # This function changes the parameter and state

    if 'rotational_step' not in state:
        # Separate step variable in case step is updated elsewhere e.g. Adam
        state['rotational_step'] = 0
    state['rotational_step'] += 1

    if 'norm' not in state:
        # Save initial weight norm
        if group['zero_mean']:
            p_zero = zero_mean(p, group['per_neuron'])
            state['norm'] = tensor_norm(p_zero, group['per_neuron'])
        else:
            state['norm'] = tensor_norm(p, group['per_neuron'])

    # Project the update, remove mean and radial components from it
    if group['zero_mean']:
        d_p = zero_mean(d_p, group['per_neuron'])
    d_p = d_p - p * (dot_product(d_p, p, group['per_neuron']) / state['norm']**2)

    # Keep track of the average update size to use to control the average rotation
    undf = group['update_norm_decay_factor']
    d_p_norm2 = (tensor_norm(d_p, group['per_neuron'])**2).detach()
    state['update_norm2'] = (1 - undf) * d_p_norm2 + undf * state.get('update_norm2',0)
    avg_update_norm = torch.sqrt(state['update_norm2'] / (1 - undf**state['rotational_step']))

    # Perform angular update with the given expected avg_rotation
    roteps = group['rotational_eps']
    p_new = p - avg_rotation * (d_p / (roteps + avg_update_norm)) * state['norm']
    p_new = p_new * state['norm'] / tensor_norm(p_new, group['per_neuron'])
    p.copy_(p_new)
