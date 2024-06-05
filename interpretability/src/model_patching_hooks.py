from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
import torch


def token_head_patching_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head: int,
    cache: Float[torch.Tensor, "batch pos head_index d_head"],
    token_position: int
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    # Each HookPoint has a name attribute giving the name of the hook.
    value[:, token_position, head, :] = cache[hook.name][head, :]
    return value