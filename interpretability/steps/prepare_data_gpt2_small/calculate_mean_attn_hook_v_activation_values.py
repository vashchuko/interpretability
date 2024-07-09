import os
import sys
import json
import pickle
import dvc.api
import torch

from src.generate_mean_activation_values import generate_mean_attn_activation_values

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 train.py split_directory\n'
    )
    sys.exit(1)

model_file = sys.argv[1]
abc_examples_file = sys.argv[2]
output_folder = sys.argv[3]

with open(model_file, 'rb') as f:
    model = pickle.load(f)

with open(abc_examples_file, 'r') as f:
  abc_examples = json.load(f)

params = dvc.api.params_show()
keys = params['calculate_mean_activation_values']['attn_keys_hook_v']
prompt_key = params['calculate_mean_activation_values']['prompt_key']
template_words = params['calculate_mean_activation_values']['template_words']

with torch.no_grad():
    mean_activations = generate_mean_attn_activation_values(model, \
        keys, abc_examples, prompt_key, template_words)

for key in mean_activations.keys():
    with open(os.path.join(output_folder, f"mean_attn_hook_v_activation_values_word_position_{key}.pkl"), 'wb') as f:
        pickle.dump(mean_activations[key], f)