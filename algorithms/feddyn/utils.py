__all__ = ["detached_statedict"]


def detached_statedict(state_dict):
    """
    Returns copied state_dict, with clone-detached tensors
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v.clone().detach()
        new_state_dict[k].requires_grad = v.requires_grad

    return new_state_dict
