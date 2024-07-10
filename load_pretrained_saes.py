# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

# %%

from sae import Sae

# %%

import huggingface_hub

# %%

repo_id = "EleutherAI/sae-llama-3-8b-32x"

# %%
path = huggingface_hub.snapshot_download(repo_id)

# %%

path

# %%

from safetensors.torch import load_model

# %%

saes = Sae.load_from_hub(repo_id, layer=12, device='cpu')

# %%

path
# %%

import json
import os

sae_layer_path = os.path.join(path, f"layers.{12}")

with open(os.path.join(sae_layer_path, "cfg.json"), "r") as f:
    cfg_dict = json.load(f)

# %%

cfg_dict
# %%