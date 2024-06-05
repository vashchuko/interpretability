from typing import List, Callable
import tqdm
import torch
from functools import partial
import transformer_lens.utils as utils
import math
from src.model_patching_hooks import token_head_patching_hook


def activation_patching_for_attention_heads(model, 
                                            examples: List[dict], 
                                            prompt_key: str,
                                            first_token_fn: Callable[[dict], str],
                                            second_token_fn: Callable[[dict], str],
                                            mean_activation_values: torch.Tensor,
                                            token_position: int = -1):
    torch.set_grad_enabled(False)
    patching_result = {}
    patching_result['logit_diff'] = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), dtype=torch.double, device=model.cfg.device)
    patching_result['first_token_logit'] = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), dtype=torch.double, device=model.cfg.device)
    patching_result['second_token_logit'] = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), dtype=torch.double, device=model.cfg.device)

    for example in tqdm.tqdm(examples, total=len(examples)):
        prompt_tokens = model.to_tokens(example[prompt_key])

        first_token_embeddings = model.to_tokens(" " + first_token_fn(example))
        second_token_embeddings = model.to_tokens(" " + second_token_fn(example))
        
        first_token = model.to_single_str_token(first_token_embeddings[0][1].item())
        second_token = model.to_single_str_token(second_token_embeddings[0][1].item())

        logits = model(prompt_tokens)
        logit_diff = logits_to_logit_diff(model, logits, first_token, second_token)

        first_token_index = first_token_embeddings[0][1].item()
        first_token_logit = logits[0, -1, first_token_index]

        second_token_index = second_token_embeddings[0][1].item()
        second_token_logit = logits[0, -1, second_token_index]

        for layer in range(model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                # Use functools.partial to create a temporary hook function with the head fixed
                temp_hook_fn = partial(token_head_patching_hook, head=head, cache=mean_activation_values, token_position=token_position)
                # Run the model with the patching hook
                patched_logits = model.run_with_hooks(prompt_tokens, fwd_hooks=[(
                    utils.get_act_name("v", layer),
                    temp_hook_fn
                    )])
                # Calculate the logit difference
                patched_logit_diff = logits_to_logit_diff(model, patched_logits, first_token, second_token).detach()
                patched_first_token_logit = patched_logits[0, -1, first_token_index]
                patched_second_token_logit = patched_logits[0, -1, second_token_index]

                if not math.isnan(1.0 - patched_logit_diff/logit_diff):
                    patching_result['logit_diff'][layer, head] += (1.0 - patched_logit_diff/logit_diff)
                if not math.isnan(1.0 - patched_first_token_logit/first_token_logit):
                    patching_result['first_token_logit'][layer, head] += (1.0 - patched_first_token_logit/first_token_logit)
                if not math.isnan(1.0 - patched_second_token_logit/second_token_logit):
                    patching_result['second_token_logit'][layer, head] += (1.0 - patched_second_token_logit/second_token_logit)

    return patching_result


def logits_to_logit_diff(model, logits, first_token, second_token):
    first_token_index = model.to_single_token(first_token)
    second_token_index = model.to_single_token(second_token)
    return logits[0, -1, first_token_index] - logits[0, -1, second_token_index]

