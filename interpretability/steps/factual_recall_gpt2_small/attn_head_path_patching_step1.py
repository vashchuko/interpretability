import os
import sys
import json
import pickle
from src.dataset_values_accessors import get_country_mentioned_in_context_for_fact_prompt, get_fact_output, get_fact_prompt
from src.path_patching import PathPatchingWithExternalCache
from src.visualization import imshow
import torch
import dvc.api


if len(sys.argv) != 5:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 train.py split_directory\n'
    )
    sys.exit(1)

model_file = sys.argv[1]
examples_file = sys.argv[2]
mean_attn_hook_v_activation_values_file = sys.argv[3]
output_folder = sys.argv[4]

with open(model_file, 'rb') as f:
    model = pickle.load(f)

with open(examples_file, 'r') as f:
    examples = json.load(f)

with open(mean_attn_hook_v_activation_values_file, 'rb') as f:
    mean_attn_hook_v_activation_values = pickle.load(f)

params = dvc.api.params_show()
prompt_key = params['attn_head_activation_patching_on_last_token']['prompt_key']

for key in mean_attn_hook_v_activation_values.keys():
    mean_attn_hook_v_activation_values[key] = torch.from_numpy(mean_attn_hook_v_activation_values[key])

target_layer = 6
target_head = 6
target_token_position = -1
path_patching = PathPatchingWithExternalCache(model=model,
                                              target_layer=target_layer,
                                              target_head=target_head,
                                              target_token_position=target_token_position)

with torch.no_grad():
    patching_result = path_patching.run(examples=examples, prompt_access_fn=get_fact_prompt, expected_token_fn=get_fact_output, 
                    base_token_fn=get_country_mentioned_in_context_for_fact_prompt,
                    external_cache=mean_attn_hook_v_activation_values)

for key in patching_result.keys():
    imshow(patching_result[key], os.path.join(output_folder, f"attn_heads_hook_v_path_patching_last_token_target_{key}_L{target_layer}H{target_head}.html"),\
        xaxis="Head", yaxis="Layer", title=f"Cumulative impact on {str.replace(key, '_', ' ')} through target head")

    with open(os.path.join(output_folder, f"attn_heads_hook_v_path_patching_last_token_target_{key}_L{target_layer}H{target_head}.pkl"), 'wb') as f:
        pickle.dump(patching_result, f)

