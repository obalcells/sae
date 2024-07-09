"""Tools for tokenizing and manipulating text datasets."""

import math
from multiprocessing import cpu_count
from typing import TypeVar, Union
from jaxtyping import Int, Float
from torch import Tensor
import heapq
from typing import List, Tuple, Dict
import time
import random

import torch
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase, AutoTokenizer

T = TypeVar("T", bound=Union[Dataset, DatasetDict])


def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = cpu_count() // 2,
    text_key: str = "text",
    max_seq_len: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
) -> T:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_seq_len` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_seq_len: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.

    Returns:
        The chunked and tokenized dataset.
    """

    def _tokenize_fn(x: dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        sep = tokenizer.eos_token or "<|endoftext|>"
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output.input_ids[0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size]
                for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        return output

    data = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    return data.with_format(format, columns=["input_ids"])


CHAT_TEMPLATE_DICT = {
    "Meta-Llama-3-8B-Instruct": """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
}

def apply_chat_template(tokenizer: AutoTokenizer, instruction: str, output: str, remove_bos_token: bool = False, template_name: str = 'Meta-Llama-3-8B-Instruct') -> str:
    if remove_bos_token:
        return CHAT_TEMPLATE_DICT[template_name].format(instruction=instruction).replace(tokenizer.bos_token, '') + output
    else:
        return CHAT_TEMPLATE_DICT[template_name].format(instruction=instruction) + output

def chunk_and_tokenize_chat_slow(
    data: T,
    tokenizer: AutoTokenizer,
    *,
    format: str = "torch",
    num_proc: int = cpu_count() // 2,
    instruction_key: str = 'instruction',
    output_key: str = 'output',
    max_seq_len: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
    batch_size: int = 2048,
    remove_bos_token: bool = False,
    max_output_len: int = 1024,
    batch_size_multiplier: int = 2,
) -> T:

    chunk_size = min(tokenizer.model_max_length, max_seq_len)
    sep_token_id = tokenizer.bos_token_id
    template_name = tokenizer.name_or_path.split("/")[-1]

    def _tokenize_fn(x: dict[str, list]):
        instructions = x[instruction_key]
        outputs = x[output_key]

        prompts = [
            apply_chat_template(tokenizer=tokenizer, instruction=instruction, output=output[:max_output_len], template_name=template_name, remove_bos_token=remove_bos_token)
            for instruction, output in zip(instructions, outputs)
        ]
        input_ids = [tokenizer.encode(prompt)[1:] for prompt in prompts]

        if sum(len(ids) for ids in input_ids) < chunk_size:
            raise ValueError(f"Not enough data to create a single complete batch. Sum of input ids lengths is: {sum(len(ids) for ids in input_ids)}")

        chunks: Int[Tensor, 'batch_size pos'] = pack_conversations(
            input_ids, sep_token_id=sep_token_id, add_sep_token=False, target_chunk_size=chunk_size, batch_size=batch_size, verbose=False
        )
        output = { "input_ids": chunks }
        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            sum_lengths = sum(len(ids) for ids in input_ids)
            print(f"Sum of input ids is: {sum_lengths}")
            raise ValueError(
                f"Not enough data to create a single complete batch. Sum of input ids lengths is: {sum_lengths}"
            )
        return output

    def _get_token_length(example):
        token_length = len(tokenizer.encode(
            apply_chat_template(tokenizer=tokenizer, instruction=example['instruction'], output=example['output'][:max_output_len], template_name=template_name, remove_bos_token=remove_bos_token)
        ))
        return token_length

    # Efficiently sample 100 random elements from the dataset
    random.seed(0)
    random_indices = random.sample(range(len(data)), min(100, len(data)))
    sampled_data = data.select(random_indices)

    # Calculate token lengths for the sampled data
    avg_token_length = sum([_get_token_length(example) for example in sampled_data]) / len(sampled_data)
    
    # Calculate the optimal batch size
    optimal_batch_size = max(1, int(batch_size_multiplier * batch_size * (chunk_size / avg_token_length)))
    
    print(f"Average token length: {avg_token_length:.2f}")
    print(f"Optimal batch size: {optimal_batch_size}")

    data = data.map(
        _tokenize_fn,
        batched=True,
        batch_size=optimal_batch_size,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    return data.with_format(format, columns=["input_ids"])


def pack_conversations(input_ids: List[List[int]], sep_token_id: int=None, add_sep_token: bool=False, target_chunk_size: int = 1024, batch_size: int = 8, verbose: bool = True) -> Int[Tensor, 'batch_size pos']:
    """Pack conversations into a single tensor."""
    sorted_input_ids = sorted(input_ids, key=len, reverse=True)

    # initialize 'batch_size' groups
    chunks = [(0, [])]
    num_prompts_included = 0
    
    heapq.heapify(chunks)

    i = 0
    while i < len(sorted_input_ids):
        chunk_size, chunk = heapq.heappop(chunks)

        if chunk_size + len(sorted_input_ids[i]) > target_chunk_size:
            if chunk_size > 0 and len(chunks) < batch_size:
                # we add the previously popped element back into the heap
                heapq.heappush(chunks, (chunk_size, chunk))
                chunk_size = 0
                chunk = []

        if chunk_size + len(sorted_input_ids[i]) > target_chunk_size:
            sorted_input_ids[i] = sorted_input_ids[i][:target_chunk_size-chunk_size]

        chunk += sorted_input_ids[i]
        if add_sep_token and chunk_size > 0 and sep_token_id is not None:
            chunk += [sep_token_id]

        heapq.heappush(chunks, (len(chunk), chunk))
        num_prompts_included += 1

        i += 1

    if verbose:
        # Get the minimum non-zero chunk size
        min_chunk_size = min(size for size, _ in chunks if size > 0)

        # Calculate the total number of tokens before pruning
        total_tokens = sum(chunk_size for chunk_size, chunk in chunks if chunk_size > 0)
        # Calculate the number of tokens after pruning
        pruned_tokens = sum(chunk_size - min_chunk_size for chunk_size, chunk in chunks if chunk_size > 0)
        # Calculate the percentage of pruned tokens
        pruned_percentage = pruned_tokens / total_tokens * 100 if total_tokens > 0 else 0

        print(f"Number of unused elements: {(len(input_ids) - num_prompts_included) / len(input_ids) * 100:.2f}%")
        print(f"Percentage of pruned tokens: {pruned_percentage:.2f}%")

    if not any(chunk_size >= target_chunk_size for chunk_size, _ in chunks):
        if batch_size == 1:
            raise ValueError(f"There aren't enough tokens to create a single batch of size {target_chunk_size}")

        # try again with a smaller batch size
        return pack_conversations(
            input_ids,
            sep_token_id=sep_token_id,
            add_sep_token=add_sep_token,
            target_chunk_size=target_chunk_size,
            batch_size=batch_size // 2,
            verbose=verbose
        )

    # due to the separation tokens, some chunks might be larger than `target_chunk_size`
    chunks: List[List[int]] = [chunk[:target_chunk_size] for chunk_size, chunk in chunks if chunk_size == target_chunk_size]

    input_ids = torch.stack([torch.tensor(chunk) for chunk in chunks])

    return input_ids

def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> list[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names
