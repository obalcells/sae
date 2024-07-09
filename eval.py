import os
import gc
from typing import List, Dict
import einops
import torch
from torch import Tensor
from jaxtyping import Int, Float
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset

from sae import Sae
from sae.hook_utils import add_hooks, get_sae_hooks
from sae.utils import calculate_kl_divergence, calculate_ce_loss
from sae.hook_utils import add_hooks, get_sae_hooks
from sae.data import chunk_and_tokenize_chat_slow 

def evaluate_loss(model: AutoTokenizer, dataset: Dataset, fwd_hooks=[], fwd_pre_hooks=[], batch_size: int = 8, n_batches: int = 128, verbose: bool = True):
    accumulated_baseline_loss = torch.tensor(0, dtype=torch.float64, device=model.device)
    accumulated_hooked_sae_loss = torch.tensor(0, dtype=torch.float64, device=model.device)
    accumulated_n_tokens = torch.tensor(0, dtype=torch.int64, device=model.device)

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    pbar = tqdm(dl, desc="Loss Evaluation", disable=not verbose, total=n_batches if n_batches else len(dl))

    for batch_idx, batch in enumerate(pbar):
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != "loss_mask"}
        loss_mask = batch["loss_mask"].to(model.device) if "loss_mask" in batch else torch.ones_like(inputs["input_ids"], device=model.device)

        accumulated_baseline_loss += calculate_ce_loss(
            model(**inputs).logits,
            inputs["input_ids"],
            loss_mask
        )

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks, **inputs):
            accumulated_hooked_sae_loss += calculate_ce_loss(
                model(**inputs).logits,
                inputs["input_ids"],
                loss_mask
            )

        accumulated_n_tokens += loss_mask.sum()

        # Clear cache and delete tensors to free up memory
        del inputs, loss_mask
        gc.collect()
        torch.cuda.empty_cache()

        if batch_idx >= n_batches:
            break

    avg_baseline_loss = accumulated_baseline_loss / accumulated_n_tokens
    avg_hooked_sae_loss = accumulated_hooked_sae_loss / accumulated_n_tokens

    relative_diff = (avg_hooked_sae_loss - avg_baseline_loss) / avg_baseline_loss

    if verbose:
        print(f"Average Baseline Loss: {avg_baseline_loss:.4f}")
        print(f"Average Hooked SAE Loss: {avg_hooked_sae_loss:.4f}")
        print(f"Relative Difference: {relative_diff:.4%}")
        print(f"Number of Tokens: {accumulated_n_tokens.item():,}")

    return avg_baseline_loss, avg_hooked_sae_loss, relative_diff, accumulated_n_tokens

@torch.no_grad()
def evaluate_kl_div(model: AutoTokenizer, dataset: Dataset, fwd_hooks=[], fwd_pre_hooks=[], batch_size: int = 8, n_batches: int = 128, verbose: bool = True):
    accumulated_kl_div = torch.tensor(0, dtype=torch.float64, device=model.device)
    accumulated_n_tokens = torch.tensor(0, dtype=torch.int64, device=model.device)

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    pbar = tqdm(dl, desc="KL Div Evaluation", disable=not verbose, total=n_batches if n_batches else len(dl))

    for batch_idx, batch in enumerate(pbar):
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != "loss_mask"}
        loss_mask = batch["loss_mask"].to(model.device) if "loss_mask" in batch else torch.ones_like(inputs["input_ids"], device=model.device)

        baseline_logits = model(**inputs).logits.cpu()

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks, **batch):
            intervention_logits = model(**inputs).logits.cpu()

        # Compute KL divergence between baseline and SAE logits
        kl_div_batch = calculate_kl_divergence(baseline_logits, intervention_logits)

        # Apply loss mask to KL divergence
        # masked_kl_div = kl_div_batch * loss_mask
        masked_kl_div = kl_div_batch
        
        # Accumulate KL divergence and token count
        accumulated_kl_div += masked_kl_div.sum().cuda()
        accumulated_n_tokens += loss_mask.sum().cuda()

        if batch_idx >= n_batches:
            break

        del baseline_logits, intervention_logits
        gc.collect()
        torch.cuda.empty_cache()

    # Compute average KL divergence
    avg_kl_div = accumulated_kl_div / accumulated_n_tokens

    if verbose:
        print(f"Average KL Divergence: {avg_kl_div:.4f}")
        print(f"Number of Tokens: {accumulated_n_tokens.item():,}")

    return avg_kl_div.item(), accumulated_n_tokens.item()

def evaluate_generations(model: AutoTokenizer, dataset: Dataset, tokenizer: AutoTokenizer, fwd_hooks=[], fwd_pre_hooks=[], batch_size: int = 8, n_batches: int = 128, generation_config: GenerationConfig = None, max_tokens: int = 32, do_sample: bool = False, verbose: bool = True):

    if generation_config is None:
        generation_config = GenerationConfig(max_new_tokens=max_tokens, do_sample=do_sample)
        generation_config.pad_token_id = model.pad_token_id

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    pbar = tqdm(dl, desc="Generations", disable=not verbose, total=n_batches if n_batches else len(dl))

    prompts = []
    baseline_generations = []
    hooked_sae_generations = []

    for batch_idx, batch in enumerate(pbar):

        batch_size, seq_len = batch["input_ids"].shape

        prompts.extend([
            tokenizer.decode(batch["input_ids"][batch_idx, (batch["attention_mask"][batch_idx] == 1)], skip_special_tokens=False).strip()
            for batch_idx in range(batch_size)
        ])

        baseline_generation_toks = model.generate(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            generation_config=generation_config,
        )[:, seq_len:]

        for generation_idx, generation in enumerate(baseline_generation_toks):
            baseline_generations.append(model.tokenizer.decode(generation, skip_special_tokens=True).strip())

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks, **batch):
            generation_toks = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                generation_config=generation_config,
            )[:, seq_len:]

            for generation_idx, generation in enumerate(generation_toks):
                hooked_sae_generations.append(model.tokenizer.decode(generation, skip_special_tokens=True).strip())

        if batch_idx >= n_batches:
            break

    if verbose:
        for i in range(3):
            print(f"Prompt: {repr(prompts[i])}")
            print(f"Baseline Generation: {repr(baseline_generations[i])}")
            print(f"Hooked SAE Generation: {repr(hooked_sae_generations[i])}")

    results = [
        {
            "prompt": prompt,
            "baseline_generation": baseline_generation,
            "hooked_sae_generation": hooked_sae_generation,
        }
        for prompt, baseline_generation, hooked_sae_generation in zip(prompts, baseline_generations, hooked_sae_generations)
    ]

    return results

def parse_args():
    pass

def main():

    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    n_batches = 32
    batch_size = 4
    max_seq_len = 64
    dataset = "teknium/openhermes"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16,
    )

    # Load the SAE
    sae = Sae.load_from_disk(f"./checkpoints/layer_10/").to('cuda')
    sae_dict = { "layer_10": sae }

    sae_fwd_hooks, saw_fwd_pre_hooks = get_sae_hooks(
        model_block_modules=model.model.layers, sae_dict=sae_dict
    )

    # Load and tokenize dataset
    dataset = load_dataset(dataset, split="train", trust_remote_code=True)
    dataset = dataset.select(range(1000))

    tokenized = chunk_and_tokenize_chat_slow(
        dataset,
        tokenizer,
        instruction_key="instruction",
        output_key="output",
        max_seq_len=max_seq_len,
        remove_bos_token=False,
        batch_size=batch_size,
        batch_size_multiplier=10,
    )

    evaluate_kl_div(
        model=model, dataset=tokenized, fwd_pre_hooks=saw_fwd_pre_hooks, fwd_hooks=sae_fwd_hooks, n_batches=n_batches, batch_size=batch_size
    )

if __name__ == '__main__':
    main()




