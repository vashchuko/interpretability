import os
import sys
import pickle
from src.visualization import imshow
import torch
import dvc.api
from tqdm import tqdm
import re


if len(sys.argv) != 5:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 train.py split_directory\n'
    )
    sys.exit(1)

model_file = sys.argv[1]
data_directory = sys.argv[2]
output_folder = sys.argv[3]
params_header = sys.argv[4]

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

params = dvc.api.params_show(stages=f"{params_header}")
target_layer = params[params_header]['target_layer']
target_head = params[params_header]['target_head']
target_token_position = params[params_header]['target_token_position']

directories = [d for d in os.listdir(data_directory) if d != '.gitignore']

with open(model_file, 'rb') as f:
    model = pickle.load(f)

patching_result = {}
for token_position in tqdm(range(target_token_position + 1)):
    patching_result[str(token_position)] = {}

    patching_result[str(token_position)]['logit_diff'] = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), dtype=torch.double, device=model.cfg.device)
    patching_result[str(token_position)]['expected_token_logit'] = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), dtype=torch.double, device=model.cfg.device)
    patching_result[str(token_position)]['base_token_logit'] = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), dtype=torch.double, device=model.cfg.device)

token_position_pattern = r"path_patching_token_position_\d+"
target_pattern = rf"target_\w+_L{target_layer}H{target_head}"

for directory in directories:
    files = [f for f in os.listdir(os.path.join(data_directory, directory)) if f != '.gitignore']
    for file in files:
        token_position_match = re.search(token_position_pattern, file)
        target_match = re.search(target_pattern, file)
        if token_position_match and target_match:
            token_position = token_position_match.group().split('_')[-1]
            target = target_match.group().replace('target_', '').replace(f'_L{target_layer}H{target_head}', '')

            with open(os.path.join(data_directory, directory, file), 'rb') as f:
                patching_result[str(token_position)][target] += pickle.load(f)

for key in patching_result.keys():
    for inner_key in patching_result[key].keys():
        patching_result[str(token_position)][target] /= len(directories)

for key in patching_result.keys():
    for inner_key in patching_result[key].keys():
        with open(os.path.join(output_folder, f"attn_heads_hook_v_path_patching_token_position_{key}_target_{inner_key}_L{target_layer}H{target_head}.pkl"), 'wb') as f:
            pickle.dump(patching_result[key][inner_key], f)

for key in patching_result.keys():
    for inner_key in patching_result[key].keys():
        imshow(patching_result[key][inner_key], os.path.join(output_folder, f"attn_heads_hook_v_path_patching_token_position_{key}_target_{inner_key}_L{target_layer}H{target_head}.html"),\
            xaxis="Head", yaxis="Layer", title=f"Cumulative impact on {str.replace(inner_key, '_', ' ')} through target head on token position {key}")


