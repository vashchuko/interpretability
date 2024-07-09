import os
import sys
import json
import pickle
from src.dataset_values_accessors import get_country_mentioned_in_context_for_fact_prompt, get_fact_output, get_fact_prompt
from src.path_patching import PathPatchingWithExternalCache
from src.visualization import imshow
import torch
import dvc.api


if len(sys.argv) != 7:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 train.py split_directory\n'
    )
    sys.exit(1)

model_file = sys.argv[1]
examples_file = sys.argv[2]
mean_attn_hook_v_activation_values_directory = sys.argv[3]
output_folder = sys.argv[4]
example_id = sys.argv[5]
params_header = sys.argv[6]

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

files = [f for f in os.listdir(mean_attn_hook_v_activation_values_directory) if f != '.gitignore']

mean_attn_hook_v_activation_values = {}
for file in files:
    with open(os.path.join(mean_attn_hook_v_activation_values_directory, file), 'rb') as f:
        mean_attn_hook_v_activation_values[str(file.split('.')[0].split('_')[-1])] = pickle.load(f)

with open(model_file, 'rb') as f:
    model = pickle.load(f)

with open(examples_file, 'r') as f:
    examples = json.load(f)

params = dvc.api.params_show(stages=f"{params_header}@{example_id}")
prompt_key = params['attn_head_activation_patching_on_last_token']['prompt_key']
target_layer = params[params_header]['target_layer']
target_head = params[params_header]['target_head']
target_token_position = params[params_header]['target_token_position']

for key in mean_attn_hook_v_activation_values.keys():
    for inner_key in mean_attn_hook_v_activation_values[key].keys():
        mean_attn_hook_v_activation_values[key][inner_key] = torch.from_numpy(mean_attn_hook_v_activation_values[key][inner_key])

path_patching = PathPatchingWithExternalCache(model=model,
                                              target_layer=target_layer,
                                              target_head=target_head,
                                              target_token_position=target_token_position)

with torch.no_grad():
    patching_result = path_patching.run(example=examples[int(example_id)], prompt_access_fn=get_fact_prompt, expected_token_fn=get_fact_output, 
                    base_token_fn=get_country_mentioned_in_context_for_fact_prompt,
                    external_cache=mean_attn_hook_v_activation_values)

for key in patching_result.keys():
    for inner_key in patching_result[key].keys():
        with open(os.path.join(output_folder, f"attn_heads_hook_v_path_patching_token_position_{key}_target_{inner_key}_L{target_layer}H{target_head}_example_{example_id}.pkl"), 'wb') as f:
            pickle.dump(patching_result[key][inner_key], f)


