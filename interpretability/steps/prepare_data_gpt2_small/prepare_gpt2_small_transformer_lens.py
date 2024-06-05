import sys
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer
import pickle

sys.path.append('./src/')

if len(sys.argv) != 2:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 train.py split_directory\n'
    )
    sys.exit(1)

output_file = sys.argv[1]
device = utils.get_device()
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

with open(output_file, 'wb') as f:
    pickle.dump(model, f)
