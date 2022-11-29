import torch
import torch.nn as nn

__all__ = ["flatten_weights", "flatten_grads", "assign_weights", "assign_grads"]


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


def flatten_grads(model):
    """
    Flattens the gradients of a model (after `.backward()` call) as a single, large vector.
    :param model: PyTorch model.
    :return: 1D torch Tensor
    """
    all_grads = []
    for name, param in model.named_parameters():
        all_grads.append(param.grad.view(-1))
    return torch.cat(all_grads)


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


def assign_grads(model, grads):
    """
    Similar to `assign_weights` but this time, manually assign `grads` vector to a model.
    :param model: PyTorch Model.
    :param grads: Gradient vectors.
    :return:
    """
    state_dict = model.state_dict(keep_vars=True)
    index = 0
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
        state_dict[param].grad = (
            grads[index : index + param_count].view(param_shape).clone()
        )
        index += param_count
    model.load_state_dict(state_dict)
    return model
