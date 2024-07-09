from torch import Tensor
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import Tensor
from jaxtyping import Float, Int
from torch.nn.functional import kl_div, log_softmax, softmax


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

# @torch.no_grad()
# def compute_kl_over_dataset(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, batch_iterator, n_batches=256, fwd_pre_hooks=[], fwd_hooks=[]):
#     accumulated_kl_div = torch.tensor(0, dtype=torch.float64, device=model.device)
#     accumulated_n_tokens = torch.tensor(0, dtype=torch.int64, device=model.device)

#     batch_idx = 0
#     for inputs, loss_mask in batch_iterator:
#         inputs = inputs.to(model.device)
#         loss_mask = loss_mask.to(model.device)

#         input_ids = inputs["input_ids"]

#         baseline_logits = model(**inputs).logits

#         with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
#             intervention_logits = model(**inputs).logits

#         # Compute KL divergence between baseline and SAE logits
#         kl_div_batch = get_kl_divergence(baseline_logits, intervention_logits)
        
#         # Apply loss mask to KL divergence
#         masked_kl_div = kl_div_batch * loss_mask
        
#         # Accumulate KL divergence and token count
#         accumulated_kl_div += masked_kl_div.sum()
#         accumulated_n_tokens += loss_mask.sum()

#         batch_idx += 1
#         if batch_idx >= n_batches:
#             break

#     # Compute average KL divergence
#     avg_kl_div = accumulated_kl_div / accumulated_n_tokens

#     return avg_kl_div.item(), accumulated_n_tokens.item()

def calculate_kl_divergence(original_logits, sae_logits):
    original_probs = softmax(original_logits, dim=-1)
    sae_probs = softmax(sae_logits, dim=-1)
    return kl_div(sae_probs.log(), original_probs, reduction='batchmean')

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