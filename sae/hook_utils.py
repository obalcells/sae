from typing import List, Dict, Callable, Tuple
import contextlib
import einops
from torch import Tensor
from jaxtyping import Int, Float
import torch.nn as nn
import einops
import functools
import torch

from sae import Sae

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_sae_fwd_pre_hook(sae: Sae, reconstruct_bos_token: bool = False):
    def hook_fn(module, input, input_ids: Int[Tensor, "batch_size seq_len"]=None, attention_mask: Int[Tensor, "batch_size seq_len"]=None):
        nonlocal sae, reconstruct_bos_token

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        dtype = activation.dtype
        batch_size, seq_pos, d_model = activation.shape

        reshaped_activation = einops.rearrange(activation, "b s d -> (b s) d")
        reconstructed_activation = sae(reshaped_activation).sae_out.to(dtype)
        reconstructed_activation = einops.rearrange(reconstructed_activation, "(b s) d -> b s d", b=batch_size, s=seq_pos)

        if not reconstruct_bos_token:
            if attention_mask is not None:
                # We don't want to reconstruct at the first sequence token (<|begin_of_text|>)
                bos_token_positions: Int[Tensor, 'batch_size'] = (attention_mask == 0).sum(dim=1)
                reconstructed_activation[:, bos_token_positions, :] = activation[:, bos_token_positions, :]
            elif seq_pos > 1:
                # we assume that the first token is always the <|begin_of_text|> token in case
                # the prompt contains multiple sequence positions (if seq_pos == 1 we're probably generating)
                reconstructed_activation[:, 0, :] = activation[:, 0, :]

        if isinstance(input, tuple):
            return (reconstructed_activation, *input[1:])
        else:
            return reconstructed_activation
    return hook_fn


def get_sae_hooks(model_block_modules: List[nn.Module], sae_dict: Dict[str, Sae], reconstruct_bos_token: bool = False):
    """
    Get the hooks for the SAE layers.

    args:
        model_block_modules: List[nn.Module]: the model block modules to hook
        sae_dict: Dict[str, Sae]: the SAE layers. The keys in the dictionary have the format 'layer_<layer_number>'.
        reconstruct_bos_token: bool: whether to reconstruct the <|begin_of_text|> token
    """

    fwd_hooks = []

    fwd_pre_hooks = [
        (
            model_block_modules[layer], 
            get_sae_fwd_pre_hook(sae=sae_dict[f"layer_{layer}"], reconstruct_bos_token=reconstruct_bos_token)
        )
        for layer in range(len(model_block_modules))
        if f"layer_{layer}" in sae_dict
    ]

    return fwd_hooks, fwd_pre_hooks