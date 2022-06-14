from collections import OrderedDict


def convert_ddp_state_dict(state_dict: OrderedDict) -> OrderedDict:
    """
    Convert a DDP model's state dict to normal model's state_dict.

    Args:
        state_dict: DDP model's state_dict.
    """
    if not next(iter(state_dict)).startswith('module.'):
        return state_dict

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.removeprefix('module.')
        new_state_dict[new_key] = value
    return new_state_dict


def remove_items_in_state_dict(state_dict: OrderedDict, keys_to_remove: list) -> OrderedDict:
    for key in keys_to_remove:
        state_dict.pop(key)
    return state_dict
