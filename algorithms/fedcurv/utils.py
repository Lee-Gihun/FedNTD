import torch
import numpy as np
import torch.nn as nn

__all__ = ["flatten_weights"]


def flatten_weights(model, numpy_output=True):
    """
    Flattens a PyTorch model. i.e., concat all parameters as a single, large vector.
    :param model: PyTorch model
    :param numpy_output: should the output vector be casted as numpy array?
    :return: the flattened (vectorized) model parameters either as Numpy array or Torch tensors
    """
    all_params = []
    for param in model.parameters():
        all_params.append(param.view(-1))
    all_params = torch.cat(all_params)
    if numpy_output:
        return all_params.cpu().detach().numpy()
    return all_params
