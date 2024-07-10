# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

# %%

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import time
from jaxtyping import Int, Float
from torch import Tensor
import argparse

from sae import Sae, SaeConfig, SaeTrainer, TrainConfig
from sae.data import chunk_and_tokenize, chunk_and_tokenize_chat

def parse_args():
    parser = argparse.ArgumentParser(description="SAE Training Script")
    
    # TrainConfig arguments
    parser.add_argument("--run_name", type=str, required=True, help="Name of the wandb run")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size measured in sequences")
    parser.add_argument("--grad_acc_steps", type=int, default=1, help="Number of steps over which to accumulate gradients")
    parser.add_argument("--micro_acc_steps", type=int, default=1, help="Chunk the activations into this number of microbatches for SAE training")
    parser.add_argument("--lr", type=float, default=None, help="Base LR. If None, it is automatically chosen based on the number of latents")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000, help="Number of warmup steps for learning rate")
    parser.add_argument("--auxk_alpha", type=float, default=0.0, help="Weight of the auxiliary loss term")
    parser.add_argument("--dead_feature_threshold", type=int, default=10_000_000, help="Number of tokens after which a feature is considered dead")
    parser.add_argument("--layers", type=str, default=None, help="List of layer indices to train SAEs on")
    parser.add_argument("--layer_stride", type=int, default=1, help="Stride between layers to train SAEs on")
    parser.add_argument("--distribute_layers", action="store_true", help="Store a single copy of each SAE, instead of copying them across devices")
    parser.add_argument("--save_every", type=int, default=500, help="Save SAEs every `save_every` steps")
    parser.add_argument("--log_to_wandb", action="store_true", default=True, help="Whether to log to wandb")
    parser.add_argument("--wandb_log_frequency", type=int, default=1, help="Frequency of logging to wandb")
    parser.add_argument("--load_from_checkpoint", action="store_true", default=False, help="Whether to load from the checkpoint directory")
    parser.add_argument("--load_from_hub_path", type=str, default=None, help="Path to the huggingface repo", choices=["obalcells/sae-llama-3-8b-instruct", 'EleutherAI/sae-llama-3-8b-32x'])
    parser.add_argument("--n_batches", type=int, default=None, help="Number of training batches. If None, it is automatically chosen based on the number of tokens")

    # SaeConfig arguments
    parser.add_argument("--expansion_factor", type=int, default=32, help="Multiple of the input dimension to use as the SAE dimension")
    parser.add_argument("--normalize_decoder", action="store_true", default=True, help="Normalize the decoder weights to have unit norm")
    parser.add_argument("--k", type=int, default=192, help="Number of nonzero features")
    parser.add_argument("--signed", action="store_true", default=False, help="Whether to use signed features")

    # Add new arguments for max_seq_len and model
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length for tokenization")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name or path of the pre-trained model to use")
    parser.add_argument("--dataset", type=str, default="teknium/openhermes", help="Name or path of the dataset to use", choices=['togethercomputer/RedPajama-Data-1T-Sample', 'teknium/openhermes', 'tatsu-lab/alpaca', 'HuggingFaceFW/fineweb'])
    parser.add_argument("--num_dataset_samples", type=int, default=None, help="Number of samples to use from the dataset")

    args = parser.parse_args()

    args.layers = [int(layer) for layer in args.layers.split(',')] if args.layers else None

    return args


def main():
    args = parse_args()
    
    # Initialize configs
    sae_config = SaeConfig(
        expansion_factor=args.expansion_factor,
        normalize_decoder=args.normalize_decoder,
        k=args.k,
        signed=args.signed,
    )
    
    train_config = TrainConfig(
        sae_config,
        grad_acc_steps=args.grad_acc_steps,
        micro_acc_steps=args.micro_acc_steps,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        auxk_alpha=args.auxk_alpha,
        dead_feature_threshold=args.dead_feature_threshold,
        layers=args.layers,
        layer_stride=args.layer_stride,
        distribute_layers=args.distribute_layers,
        save_every=args.save_every,
        log_to_wandb=args.log_to_wandb,
        run_name=args.run_name,
        wandb_log_frequency=args.wandb_log_frequency,
        load_from_checkpoint=args.load_from_checkpoint,
        load_from_hub_path=args.load_from_hub_path,
        n_batches=args.n_batches,
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load and tokenize dataset
    split = "train"
    if args.num_dataset_samples is not None:
        split = f"train[:{args.num_dataset_samples}]"
    dataset = load_dataset(args.dataset, split=split, trust_remote_code=True)

    dataset_type = 'chat' if args.dataset in ['teknium/openhermes', 'tatsu-lab/alpaca'] else 'text'

    if dataset_type == 'chat':
        tokenized = chunk_and_tokenize_chat(
            dataset,
            tokenizer,
            instruction_key="instruction",
            output_key="output",
            max_seq_len=args.max_seq_len,
            remove_bos_token=False,
            batch_size=args.batch_size,
        )
    elif dataset_type == 'text':
        tokenized = chunk_and_tokenize(
            dataset,
            tokenizer,
            max_seq_len=args.max_seq_len,
        )
    
    # Initialize trainer and start training
    trainer = SaeTrainer(train_config, tokenized, model)
    trainer.fit()

if __name__ == "__main__":
    main()
