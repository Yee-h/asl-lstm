from __future__ import annotations

from collections.abc import Iterable

import torch


def average_state_dicts(state_dicts: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Average multiple state_dict objects key-wise on CPU float tensors."""
    dict_list = list(state_dicts)
    if not dict_list:
        raise ValueError("state_dicts 不能为空")

    keys = list(dict_list[0].keys())
    averaged = {k: dict_list[0][k].detach().cpu().clone().float() for k in keys}

    for state in dict_list[1:]:
        if list(state.keys()) != keys:
            raise ValueError("state_dict 键不一致，无法平均")
        for key in keys:
            averaged[key] += state[key].detach().cpu().float()

    count = float(len(dict_list))
    for key in keys:
        averaged[key] /= count

    return averaged
