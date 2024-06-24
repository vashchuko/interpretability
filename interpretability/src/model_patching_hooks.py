from typing import Dict, List
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
import torch


def head_patching_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head: int,
    patching_values: Float[torch.Tensor, "head_index d_head"],
    token_position: int
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    # Each HookPoint has a name attribute giving the name of the hook.
    value[:, token_position, head, :] = patching_values[hook.name][head, :]
    return value

def residual_stream_patching_hook(
    resid: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    patching_values: Float[torch.Tensor, "batch pos d_model"],
    token_position: int
) -> Float[torch.Tensor, "batch pos d_model"]:
    resid[:, token_position, :] = patching_values[hook.name][:, token_position, :]
    return resid

def heads_patching_with_heads_exclusion_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    patching_values: Float[torch.Tensor, "head_index d_head"],
    heads_to_exclude: List[Dict[str, int]],
    token_position: int,
    n_heads: int
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    heads = [head for head in heads_to_exclude if head['layer'] == hook.layer()]

    if len(heads) == 2:
        for head in range(n_heads):
            if (heads[0]['head'] != head and heads[1]['head'] != head):
                value[:, token_position, head, :] = patching_values[hook.name][:, token_position, head, :],
    elif len(heads) == 1:
        value[:, token_position, :heads[0]['head'], :] = patching_values[hook.name][:, token_position, :heads[0]['head'], :]
        value[:, token_position, heads[0]['head'] + 1:, :] = patching_values[hook.name][:, token_position, heads[0]['head'] + 1:, :]
    else:
        value[:, token_position, :, :] = patching_values[hook.name][:, token_position, :, :]
    return value