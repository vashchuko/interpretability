

from functools import partial
from typing import Callable, List
from src.activation_patching import logits_to_logit_diff, token_logit
from src.model_patching_hooks import head_patching_hook, heads_patching_with_heads_exclusion_hook, residual_stream_patching_hook
from tqdm import tqdm
import transformer_lens.utils as utils
import torch
from jaxtyping import Float


class PathPatchingABC:

    def __init__(self,
                model,
                target_layer: int,
                target_head: int,
                target_token_position: int,
                patched_head_activation_value_name: str = 'v',
                other_heads_activation_value_name: str = 'v'):
        self.model = model
        self.target_layer = target_layer
        self.target_head = target_head
        self.target_token_position = target_token_position
        self.patched_head_activation_value_name = patched_head_activation_value_name
        self.other_heads_activation_value_name = other_heads_activation_value_name


    def get_layers_number(self, token_position):
        return self.target_layer if self.target_token_position == token_position else self.model.cfg.n_layers

    def forward_run_a(self, 
                      prompt: str, 
                      expected_token: str, 
                      base_token: str):
        logits, cache = self.model.run_with_cache(prompt)

        logit_diff = logits_to_logit_diff(self.model, logits, first_token=expected_token, second_token=base_token)

        expected_token_logit = token_logit(self.model, logits, expected_token)
        base_token_logit = token_logit(self.model, logits, base_token)

        return cache, logit_diff, expected_token_logit, base_token_logit

    def forward_run_b(self, prompt: str):
        _, abc_cache = self.model.run_with_cache(prompt)
        return abc_cache

    def forward_run_c(self, 
                      prompt: str, 
                      run_a_cache: Float[torch.Tensor, "batch_index token_index head_index d_head"], 
                      run_b_cache: Float[torch.Tensor, "head_index d_head"], 
                      token_position: int = -1):
        cache_dict_for_target_head = {}

        for layer in range(self.get_layers_number(token_position)):
            for head in range(self.model.cfg.n_heads):

                model_fwd_hooks = [
                        (utils.get_act_name(self.other_heads_activation_value_name, inner_layer),
                            partial(heads_patching_with_heads_exclusion_hook,
                                    patching_values=run_a_cache,
                                    heads_to_exclude=[{ 'layer': layer, 'head': head}], # , { 'layer': self.target_layer, 'head': self.target_head }
                                    token_position=token_position,
                                    n_heads=self.model.cfg.n_heads)
                        ) for inner_layer in range(self.get_layers_number(token_position))
                    ]

                model_fwd_hooks.append(
                    (utils.get_act_name(self.patched_head_activation_value_name, layer),
                    partial(head_patching_hook,
                            patching_values=run_b_cache,
                            head=head,
                            token_position=token_position)
                    ))

                # Run the model with the patching hook
                cache_dict = self.model.add_caching_hooks(names_filter=[f'blocks.{self.target_layer}.attn.hook_{self.patched_head_activation_value_name}'])
                _ = self.model.run_with_hooks(prompt, fwd_hooks=model_fwd_hooks)
                cache_dict_for_target_head[f'{layer}_{head}_{token_position}'] = {}
                cache_dict_for_target_head[f'{layer}_{head}_{token_position}'][f'blocks.{self.target_layer}.attn.hook_{self.other_heads_activation_value_name}'] = cache_dict[f'blocks.{self.target_layer}.attn.hook_{self.other_heads_activation_value_name}'][-1, token_position, :, :].detach().clone()
        return cache_dict_for_target_head

    def forward_run_d(self, 
                      prompt: str, 
                      expected_token: str, 
                      base_token: str, 
                      target_head_cache: dict[str, Float[torch.Tensor, "head_index d_head"]], 
                      run_a_cache: Float[torch.Tensor, "head_index d_head"], 
                      run_a_logit_diff: Float, 
                      token_position: int = -1,
                      run_a_expected_token_logit: Float = None, 
                      run_a_base_token_logit: Float = None):
        patching_result = {}

        patching_result['logit_diff'] = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads), dtype=torch.double, device=self.model.cfg.device)
        patching_result['expected_token_logit'] = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads), dtype=torch.double, device=self.model.cfg.device)
        patching_result['base_token_logit'] = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads), dtype=torch.double, device=self.model.cfg.device)

        for layer in range(self.get_layers_number(token_position)):
            for head in range(self.model.cfg.n_heads):

                model_fwd_hooks = [(utils.get_act_name(self.other_heads_activation_value_name, inner_layer), 
                                    partial(heads_patching_with_heads_exclusion_hook, 
                                        patching_values=run_a_cache, 
                                        heads_to_exclude=[{ 'layer': self.target_layer, 'head': self.target_head }],
                                        token_position=token_position,
                                        n_heads=self.model.cfg.n_heads)) for inner_layer \
                                            in range(self.get_layers_number(token_position))]

                model_fwd_hooks.append((
                    utils.get_act_name(self.other_heads_activation_value_name, self.target_layer),
                    partial(head_patching_hook,
                            patching_values=target_head_cache[f'{layer}_{head}_{token_position}'],
                            head=self.target_head,
                            token_position=token_position)
                    ))

                model_fwd_hooks.extend(
                    [
                        (utils.get_act_name("resid_post", inner_layer),
                        partial(residual_stream_patching_hook,
                                patching_values=run_a_cache,
                                token_position=token_position)
                        )
                        for inner_layer in range(self.get_layers_number(token_position))
                    ])

                # Run the model with the patching hook
                patched_logits = self.model.run_with_hooks(prompt, fwd_hooks=model_fwd_hooks)

                # Calculate the logit difference
                patched_logit_diff = logits_to_logit_diff(self.model, patched_logits, first_token=expected_token, second_token=base_token).detach()

                # print(f"Patched logit difference: {patched_logit_diff.item():.6f}")
                patching_result['logit_diff'][layer, head] = (1.0 - patched_logit_diff/run_a_logit_diff)
                if (run_a_expected_token_logit is not None and run_a_base_token_logit is not None):
                    patched_expected_token_logit = token_logit(self.model, patched_logits, expected_token)
                    patched_base_token_logit = token_logit(self.model, patched_logits, base_token)
                    
                    patching_result['expected_token_logit'][layer, head] = (1.0 - patched_expected_token_logit/run_a_expected_token_logit)
                    patching_result['base_token_logit'][layer, head] = (1.0 - patched_base_token_logit/run_a_base_token_logit)

        return patching_result
    

class PathPatchingWithExternalCache(PathPatchingABC):

    def __init__(self,
                model,
                target_layer: int,
                target_head: int,
                target_token_position: int,
                patched_head_activation_value_name: str = 'v',
                other_heads_activation_value_name: str = 'v'):
        super(PathPatchingWithExternalCache, self).__init__(model=model,
                                                            target_layer=target_layer,
                                                            target_head=target_head,
                                                            target_token_position=target_token_position,
                                                            patched_head_activation_value_name=patched_head_activation_value_name,
                                                            other_heads_activation_value_name=other_heads_activation_value_name)
    
    def run(self,
            examples: List[dict],
            prompt_access_fn: Callable[[dict], str],
            expected_token_fn: Callable[[dict], str],
            base_token_fn: Callable[[dict], str],
            external_cache: Float[torch.Tensor, "head_index d_head"]):
        patching_result = {}

        patching_result['logit_diff'] = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads), dtype=torch.double, device=self.model.cfg.device)
        patching_result['expected_token_logit'] = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads), dtype=torch.double, device=self.model.cfg.device)
        patching_result['base_token_logit'] = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads), dtype=torch.double, device=self.model.cfg.device)

        for example in tqdm(examples):
            run_a_prompt = self.model.to_tokens(prompt_access_fn(example))
            expected_token = self.model.to_single_str_token(self.model.to_tokens(" " + expected_token_fn(example))[0][1].item())
            base_token = self.model.to_single_str_token(self.model.to_tokens(" " + base_token_fn(example))[0][1].item())

            run_a_cache, run_a_logit_diff, run_a_expected_token_logit, run_a_base_token_logit = \
                self.forward_run_a(prompt=run_a_prompt, expected_token=expected_token, base_token=base_token)
            
            target_head_cache = \
                self.forward_run_c(prompt=run_a_prompt, run_a_cache=run_a_cache, run_b_cache=external_cache, token_position=-1)

            run_d_patching_result = self.forward_run_d(prompt=run_a_prompt, expected_token=expected_token, base_token=base_token, \
                           target_head_cache=target_head_cache, run_a_cache=run_a_cache, run_a_logit_diff=run_a_logit_diff, \
                           token_position=-1, run_a_expected_token_logit=run_a_expected_token_logit, \
                           run_a_base_token_logit=run_a_base_token_logit)
            
            for key in patching_result.keys():
                patching_result[key] += run_d_patching_result[key]

        return patching_result
