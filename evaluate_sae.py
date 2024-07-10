import os
import gc
from typing import List, Dict, Callable
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
import argparse
import json

from sae.sparse_autoencoder import Sae
from sae.hook_utils import get_sae_hooks
from sae.data import get_dataset_iterator # , chunk_and_tokenize_chat, chunk_and_tokenize, batch_iterator_chat, batch_iterator_text
from sae.utils import format_prompts
from sae.eval import evaluate_kl_div, evaluate_loss, generate_completions
from sae.config import EvaluationTaskConfig

print("Setting torch grad to False")
torch.set_grad_enabled(False)

DEFAULT_EVALUATION_TASKS: List[EvaluationTaskConfig] = [
    EvaluationTaskConfig(
        task_fn="evaluate_kl_div",
        dataset_path="teknium/openhermes",
        use_padding=False,
        chat_format=True,
        num_dataset_samples=1000,
        do_generations=False,
        max_seq_len=1024
    ),
    EvaluationTaskConfig(
        task_fn="evaluate_kl_div",
        dataset_path="togethercomputer/RedPajama-Data-1T-Sample",
        use_padding=False,
        chat_format=False,
        num_dataset_samples=1000,
        do_generations=False,
        max_seq_len=1024
    ),
    EvaluationTaskConfig(
        task_fn="evaluate_kl_div",
        dataset_path="tatsu-lab/alpaca",
        use_padding=True,
        chat_format=True,
        num_dataset_samples=1000,
        do_generations=False,
        max_seq_len=1024
    ),
    EvaluationTaskConfig(
        task_fn="evaluate_loss",
        dataset_path="teknium/openhermes",
        use_padding=False,
        chat_format=True,
        num_dataset_samples=1000,
        do_generations=False,
        max_seq_len=1024
    ),
    EvaluationTaskConfig(
        task_fn="generate_completions",
        dataset_path="tatsu-lab/alpaca",
        use_padding=True,
        chat_format=True,
        num_dataset_samples=100,
        do_generations=True,
        max_seq_len=1024
    )
]

def evaluate_sae(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae_fwd_hooks: List[Callable],
    sae_fwd_pre_hooks: List[Callable],
    batch_size: int = 8,
    tasks: List[EvaluationTaskConfig] = DEFAULT_EVALUATION_TASKS,
):

    results = []

    for task_config in tasks:
        # Generate completions
        dataset_iterator = get_dataset_iterator(
            tokenizer=tokenizer,
            task_config=task_config,
            batch_size=batch_size if task_config.task_fn != "evaluate_kl_div" else batch_size // 2, # we halve the batch size for kl div
        )

        if task_config.task_fn == "generate_completions":
            completions = generate_completions(
                model=model,
                tokenizer=tokenizer,
                batch_iterator=dataset_iterator,
                fwd_hooks=sae_fwd_hooks,
                fwd_pre_hooks=sae_fwd_pre_hooks,
            )

            results.append({
                'model_path': model.name_or_path,
                'task_config': task_config.to_dict(),
                'completions': completions,
            })

            # also store the completions locally in a json
            completions_path = f"./hooked_sae_generations/{model.name_or_path.replace('/', '_')}_{str(task_config)}_generations.json"
            os.makedirs(os.path.dirname(completions_path), exist_ok=True)
            with open(completions_path, "w") as f:
                json.dump(completions, f)

        elif task_config.task_fn == "evaluate_kl_div":

            kl_div_results = evaluate_kl_div(
                model=model,
                batch_iterator=dataset_iterator,
                fwd_hooks=sae_fwd_hooks,
                fwd_pre_hooks=sae_fwd_pre_hooks,
            )

            results.append({
                'model_path': model.name_or_path,
                'task_config': task_config.to_dict(),
                'kl_div_results': kl_div_results,
            })
        
        elif task_config.task_fn == "evaluate_loss":
            loss_results = evaluate_loss(
                model=model,
                batch_iterator=dataset_iterator,
                fwd_hooks=sae_fwd_hooks,
                fwd_pre_hooks=sae_fwd_pre_hooks,
            )

            results.append({
                'model_path': model.name_or_path,
                'task_config': task_config.to_dict(),
                'loss_results': loss_results,
            })

    return results
        

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAE on various tasks")

    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name or path of the pre-trained model to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")

    parser.add_argument("--layer", type=int, required=True, help="Layer to hook the SAE to")
    parser.add_argument("--load_from_hub_path", type=str, default="obalcells/sae-llama-3-8b-instruct", help="Path to the huggingface repo", choices=["obalcells/sae-llama-3-8b-instruct", 'EleutherAI/sae-llama-3-8b-32x'])
    parser.add_argument("--load_from_disk_path", type=str, default=None, help="Whether to load from the checkpoint directory")

    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if 'llama-3' in args.model_path.lower():
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if 'llama-3' in args.model_path.lower():
        model_block_modules = model.model.layers
    elif 'gemma-2-' in args.model_path.lower():
        raise NotImplementedError("Gemma 2 is not supported yet")
    elif 'gemma' in args.model_path.lower():
        model_block_modules = model.model.layers
    else:
        raise ValueError(f"Model {args.model_path} not supported")

    # Load SAEs
    if args.load_from_disk_path is not None:
        sae_path = f"{args.load_from_disk_path}/layer_{args.layer}"
        print(f"Loading SAE from disk: {sae_path}")
        sae = Sae.load_from_disk(sae_path).to(model.device)
    else:
        print(f"Loading SAE from hub: {args.load_from_hub_path}")
        sae = Sae.load_from_hub(args.load_from_hub_path, args.layer).to(model.device)

    sae_dict = { f"layer_{args.layer}": sae }
    sae_fwd_hooks, sae_fwd_pre_hooks = get_sae_hooks(
        model_block_modules,
        sae_dict,
        reconstruct_bos_token=False, # by default we won't reconstruct the activations at the bos token when the input is padded (not packed)
                                     # the bos token is not added to the prompts for packed text datasets
                                     # (such as the Eleuther's base SAE training dataset)
    )

    # Call evaluate_sae function
    results = evaluate_sae(
        model=model,
        tokenizer=tokenizer,
        sae_fwd_hooks=sae_fwd_hooks,
        sae_fwd_pre_hooks=sae_fwd_pre_hooks,
        batch_size=args.batch_size,
        tasks=DEFAULT_EVALUATION_TASKS,
    )

    # Remove completions from the results before printing
    for task_result in results:
        if task_result['task_config']['do_generations']:
            del task_result['completions']

    print("Results")
    print(results)

if __name__ == '__main__':
    main()




