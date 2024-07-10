from torch import Tensor
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import Tensor
from jaxtyping import Float, Int
from torch.nn.functional import kl_div, log_softmax, softmax
from typing import List, Union, Dict

@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = torch.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / torch.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if torch.norm(guess - prev) < tol:
            break

    return guess

def calculate_kl_divergence(original_logits, sae_logits, reduction='batchmean') -> Union[Float[Tensor, '1'], Float[Tensor, 'batch_size pos']]:
    original_probs = softmax(original_logits, dim=-1)
    sae_probs = softmax(sae_logits, dim=-1)

    kl_divergence = kl_div(sae_probs.log(), original_probs, reduction=reduction)

    if reduction == 'none':
        kl_divergence = kl_divergence.sum(dim=-1)

    return kl_divergence

def calculate_ce_loss(logits: Float[Tensor, 'batch_size pos d_vocab'], input_ids: Int[Tensor, 'batch_size pos'], loss_mask: Int[Tensor, 'batch_size pos']) -> Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_for_labels = log_probs[:, :-1].gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    # add a last column of zeros to log_probs_for_labels to match the shape of loss_mask
    log_probs_for_labels = torch.cat(
        [
            log_probs_for_labels,
            torch.zeros(log_probs_for_labels.shape[0]).unsqueeze(-1).to(log_probs_for_labels)
        ],
        dim=-1
    )

    # apply loss_mask
    log_probs_for_labels = log_probs_for_labels * loss_mask.to(log_probs_for_labels.device)

    return -log_probs_for_labels.sum()

CHAT_TEMPLATE_DICT = {
    "Meta-Llama-3-8B-Instruct": """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
}

def apply_chat_template(tokenizer: AutoTokenizer, instruction: str, output: str, template_name: str, remove_bos_token: bool = False) -> str:
    if remove_bos_token:
        return CHAT_TEMPLATE_DICT[template_name].format(instruction=instruction).replace(tokenizer.bos_token, '') + output
    else:
        return CHAT_TEMPLATE_DICT[template_name].format(instruction=instruction) + output

def format_prompts(tokenizer: AutoTokenizer, instructions: List[str], outputs: List[str], template_name: str = None, remove_bos_token: bool = False, max_output_len: int = None):
    if max_output_len is not None:
        outputs = [output[:max_output_len] for output in outputs]

    if template_name and template_name in CHAT_TEMPLATE_DICT:
        return [
            apply_chat_template(tokenizer=tokenizer, instruction=instruction, output=output, template_name=template_name, remove_bos_token=remove_bos_token)
            for instruction, output in zip(instructions, outputs)
        ]

    return [
        f"Instruction: {instruction}\n\nOutput: {output}"
        for instruction, output in zip(instructions, outputs)
    ]

###
# Chat dataset iterator
###
