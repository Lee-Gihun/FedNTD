import torch
import torch.nn as nn

import numpy as np

__all__ = ["update_momentum_weights", "flatten_weights", "assign_weights"]


def update_momentum_weights(m_flat, dg_flat, new_flat, beta=0.9):
    m_flat_new = beta * m_flat + (dg_flat - new_flat)
    ag_flat = dg_flat - m_flat_new  # new_flat -> dg_flat

    return m_flat_new, ag_flat


def flatten_weights(model, from_dict=False, numpy_output=True):
    """
    Flattens a PyTorch model. i.e., concat all parameters as a single, large vector.
    :param model: PyTorch model
    :param numpy_output: should the output vector be casted as numpy array?
    :return: the flattened (vectorized) model parameters either as Numpy array or Torch tensors
    """
    all_params = []

    if from_dict:
        for param in model.values():
            all_params.append((param.clone().detach()).view(-1))

    else:
        for param in model.parameters():
            all_params.append((param.clone().detach()).view(-1))
    all_params = torch.cat(all_params)

    if numpy_output:
        return all_params.cpu().clone().detach().numpy()

    return all_params


def assign_weights(model, weights):
    """
    Manually assigns `weights` of a Pytorch `model`.
    Note that weights is of vector form (i.e., 1D array or tensor).
    Usage: For implementation of Mode Connectivity SGD algorithm.
    :param model: Pytorch model.
    :param weights: A flattened (i.e., 1D) weight vector.
    :return: The `model` updated with `weights`.
    """
    state_dict = model.state_dict(keep_vars=True)
    # The index keeps track of location of current weights that is being un-flattened.
    index = 0
    # just for safety, no grads should be transferred.
    with torch.no_grad():
        for param in state_dict.keys():
            # ignore batchnorm params
            if (
                "running_mean" in param
                or "running_var" in param
                or "num_batches_tracked" in param
            ):
                continue
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] = nn.Parameter(
                torch.from_numpy(
                    weights[index : index + param_count].reshape(param_shape)
                )
            )
            index += param_count
    model.load_state_dict(state_dict)

    return model
