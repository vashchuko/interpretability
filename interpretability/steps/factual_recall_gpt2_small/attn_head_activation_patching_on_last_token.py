import os
import sys
import json
import pickle
from src.activation_patching import activation_patching_for_attention_heads
from src.dataset_values_accessors import get_country_mentioned_in_context_for_fact_prompt, get_fact_output
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
    
with torch.no_grad():
    patching_result = activation_patching_for_attention_heads(model, examples, prompt_key, \
                     get_fact_output, get_country_mentioned_in_context_for_fact_prompt, mean_attn_hook_v_activation_values, token_position=-1)

for key in patching_result.keys():
    imshow(patching_result[key], os.path.join(output_folder, f"attn_heads_hook_v_activation_patching_last_token_target_{key}.html"),\
        xaxis="Head", yaxis="Layer", title="Cumulative Impact on Logit Difference After Patching Attnetion Head")

    with open(os.path.join(output_folder, f"attn_heads_hook_v_activation_patching_last_token_target_{key}.pkl"), 'wb') as f:
        pickle.dump(patching_result[key], f)

