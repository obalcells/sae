# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

# %%
import sys
sys.path.append("../..")

# %%

import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from torch.nn.functional import kl_div, log_softmax, softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
from sae import Sae
from jaxtyping import Float, Int
from torch import Tensor
from typing import Dict, List
from colorama import Fore
import textwrap
import einops

from utils.hf_patching_utils import add_hooks
from utils.hf_models.model_base import ModelBase
from utils.hf_models.model_factory import construct_model_base
from dataset.load_data import load_triviaqa_queries

torch.set_grad_enabled(False)

# %%

def get_sae_fwd_hook(sae: Sae):
    def hook_fn(module, input, output):
        nonlocal sae

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        activation = sae(activation).sae_out

        if isinstance(input, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn

def get_all_sae_fwd_hooks(model_base: ModelBase, sae_dict: Dict[str, Sae]):

    sae_layers_dict = {layer: sae_dict[f"layer_{layer}"] for layer in range(model_base.model.config.num_hidden_layers) if f"layer_{layer}" in sae_dict}

    fwd_hooks = [
        (model_base.model_block_modules[layer], get_sae_fwd_hook(sae=sae_layers_dict[layer]))
        for layer in range(model_base.model.config.num_hidden_layers)
        if layer in sae_layers_dict
    ]

    return fwd_hooks

def print_generations(prompt, orig_generations, steered_generations):
    for i in range(len(steered_generations)):
        print("Prompt: ", repr(prompt))
        print(Fore.GREEN + f"ORIGINAL COMPLETION:")
        print(textwrap.fill(repr(orig_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(Fore.RED + f"STEERED COMPLETION:")
        print(textwrap.fill(repr(steered_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(Fore.RESET)

# %%

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_alias = "Meta-Llama-3-8B-Instruct"

model_wrapper = construct_model_base(model_name)
tokenizer = model_wrapper.tokenizer

def tokenize_instructions_fn(instructions: List[str]):
    return tokenizer(instructions, padding=True, return_tensors="pt")

model_wrapper.tokenize_instructions_fn = tokenize_instructions_fn

# %%

triviaqa_queries = load_triviaqa_queries(model_alias=model_alias, split="train", apply_chat_format=True)

# %%

sae_layer_10 = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", layer=10).to('cuda')
# %%
sae_dict = { "layer_10": sae_layer_10 }
# %%
sae_fwd_hooks = get_all_sae_fwd_hooks(model_wrapper, sae_dict)

# %%

for query_idx, query in enumerate(triviaqa_queries[:10]):

    prompt = query["question"]

    steered_generations = model_wrapper.generate_completions([prompt], fwd_hooks=sae_fwd_hooks)
    baseline_generations = model_wrapper.generate_completions([prompt], fwd_hooks=[])

    print_generations([prompt], baseline_generations, steered_generations)
    print("Correct answer:", repr(query['correct_answer']))

# %%

act = torch.rand((10, 4096), device='cuda')
sae_layer_10(act).sae_out.shape

# %%

pre_acts = sae_layer_10.pre_acts(act)
top_acts, top_indices = sae_layer_10.select_topk(pre_acts)

# %%

print("Acts shape:", top_acts.shape)
print("Indices shape:", top_indices.shape)
sae_layer_10.decode(top_acts, top_indices).shape

# %%

def decoder_matmul(sae: Sae, top_acts: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
    """
    Perform the decoder matrix multiplication, taking into account the indices of top activations.
    
    Args:
    - sae: Sae object
    - top_acts: Tensor of shape (..., k) containing the top k activations
    - top_indices: Tensor of shape (..., k) containing the indices of top k activations
    Returns:
    - Tensor of shape (..., d_in) containing the decoded output
    """
    batch_dims = top_acts.shape[:-1]
    d_sae, d_in = sae.W_dec.shape
    
    # Create a tensor of zeros with the shape of the full activation space
    full_acts = torch.zeros((*batch_dims, d_sae), device=top_acts.device, dtype=top_acts.dtype)
    
    # Scatter the top activations into their correct positions
    full_acts.scatter_(-1, top_indices, top_acts)
    
    # Perform the matrix multiplication using einops
    y = einops.einsum(full_acts, sae.W_dec, "... d_sae, d_sae d_in -> ... d_in")
    return y + sae.b_dec

# %%

random_acts = torch.rand((10, 32, 4096), device='cuda')
pre_acts = sae_layer_10.pre_acts(random_acts)
top_acts, top_indices = sae_layer_10.select_topk(pre_acts)

# %%

# Test the function
decoded_output = decoder_matmul(sae_layer_10, top_acts, top_indices)
print("Decoded output shape:", decoded_output.shape)

# Compare with the original decode method
original_decoded = sae_layer_10.decode(top_acts, top_indices)
print("Original decoded shape:", original_decoded.shape)

# Check if the results are close
print("Results are close:", torch.allclose(decoded_output, original_decoded, atol=1e-5))

# %%

sae_layer_10.W_dec.shape



# %%

print("Top acts shape:", top_acts.shape)
print("Top indices shape:", top_indices.shape)

# Decode and compute residual
sae_layer_10.decode(top_acts, top_indices).shape

# %%

sae_layer_10
# %%
