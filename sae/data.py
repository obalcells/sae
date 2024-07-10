"""Tools for tokenizing and manipulating text datasets."""

import math
from multiprocessing import cpu_count
from typing import TypeVar, Union
from jaxtyping import Int, Float
from torch import Tensor
import heapq
from typing import List, Tuple, Dict, Callable
import time
import random
import warnings
import itertools
from torch.utils.data import DataLoader

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from sae.utils import CHAT_TEMPLATE_DICT, format_prompts
from sae.config import EvaluationTaskConfig

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


def chunk_and_tokenize_chat(
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
    batch_size_multiplier: int = 4,
) -> T:

    chunk_size = min(tokenizer.model_max_length, max_seq_len)
    sep_token_id = tokenizer.bos_token_id
    template_name = tokenizer.name_or_path.split("/")[-1]

    if template_name not in CHAT_TEMPLATE_DICT:
        print(f"WARNING: Chat template for {tokenizer.name_or_path} not found. Falling back to default formatting.")
        template_name = None

    def _tokenize_fn(x: dict[str, list]):
        instructions = x[instruction_key]
        outputs = x[output_key]

        prompts = format_prompts(tokenizer, instructions, outputs, template_name=template_name, remove_bos_token=remove_bos_token, max_output_len=max_output_len)
        input_ids = [tokenizer.encode(prompt)[1:] for prompt in prompts]

        if sum(len(ids) for ids in input_ids) < chunk_size:
            warnings.warn(
                f"Not enough data to fill a batch. Duplicating prompts to reach required sequence length of {chunk_size}.",
                UserWarning
            )
            total_input_ids_len = sum(len(ids) for ids in input_ids)
            while total_input_ids_len < chunk_size:
                input_ids += input_ids
                total_input_ids_len *= 2

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
        token_length = len(
            tokenizer.encode(
                format_prompts(tokenizer, instructions=[example['instruction']], outputs=[example['output']], template_name=template_name, remove_bos_token=remove_bos_token, max_output_len=max_output_len)[0]
            )
        )
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

    # initialize a single empty chunk
    chunks = [(0, [])]
    heapq.heapify(chunks)

    i = 0
    while i < len(sorted_input_ids):
        chunk_size, chunk = heapq.heappop(chunks)

        if chunk_size >= target_chunk_size and len(chunks) + 1 < batch_size:
            # we add the chunk back into the heap and finish the loop since we can't pack more than `batch_size` chunks
            heapq.heappush(chunks, (chunk_size, chunk))
            break

        # if the next element will exceed the target chunk size, we consider creating a new empty chunk
        if chunk_size + len(sorted_input_ids[i]) > target_chunk_size and chunk_size > 0 and len(chunks) + 1 < batch_size:
            # we add the previously popped element back into the heap (so that it doesn't disappear)
            heapq.heappush(chunks, (chunk_size, chunk))

            # create a new empty chunk
            chunk_size = 0
            chunk = []

        chunk += sorted_input_ids[i]
        if add_sep_token and sep_token_id is not None:
            chunk += [sep_token_id]

        heapq.heappush(chunks, (len(chunk), chunk))

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
        # Calculate the number of prompts included
        num_prompts_included = len(input_ids) - i
        print(f"Number of unused elements: {(len(input_ids) - num_prompts_included) / len(input_ids) * 100:.2f}%")
        print(f"Percentage of pruned tokens: {pruned_percentage:.2f}%")

    if not any(chunk_size >= target_chunk_size for chunk_size, _ in chunks):
        if batch_size == 1:
            sum_of_input_id_lengths = sum(len(ids) for ids in input_ids)
            print(f"Chunk sizes: {[chunk_size for chunk_size, _ in chunks]}")
            print(f"Lengths of sorted input ids: {[len(ids) for ids in input_ids]}")
            raise ValueError(f"There aren't enough tokens to create a single batch of size {target_chunk_size}. Sum of input ids lengths is: {sum_of_input_id_lengths}")

        # try again with a smaller batch size
        return pack_conversations(
            input_ids,
            sep_token_id=sep_token_id,
            add_sep_token=add_sep_token,
            target_chunk_size=target_chunk_size,
            batch_size=batch_size // 2,
            verbose=verbose
        )

    chunks: List[List[int]] = [chunk[:target_chunk_size] for chunk_size, chunk in chunks if chunk_size >= target_chunk_size]

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

def batch_iterator_chat(repo_id: str, tokenize_instructions_fn: Callable, batch_size: int, eoi_toks=None, instruction_key='instruction', output_key='output', split='train'):
    """Yields (padded) batches from a chat dataset."""

    dataset = load_dataset(repo_id, split=split)
    dataset = dataset.shuffle(seed=42)

    # filter out instructions where the field 'input' isn't empty
    if 'input' in dataset.features:
        dataset = dataset.filter(lambda x: len(x['input'].strip()) == 0)

    dataset_instructions = dataset[instruction_key]
    dataset_outputs = dataset[output_key]

    it_instructions = iter(dataset_instructions)
    it_outputs = iter(dataset_outputs)
    while True:
        instructions_batch = list(itertools.islice(it_instructions, batch_size))
        outputs_batch = list(itertools.islice(it_outputs, batch_size))
        if not instructions_batch or not outputs_batch:
            break

        # count the number of non-completion tokens for each batch element
        instruction_toks_len = tokenize_instructions_fn(instructions=instructions_batch, outputs=[""]*len(instructions_batch))["attention_mask"].sum(dim=1)

        # we not tokenize each instruction with its corresponding completion
        inputs = tokenize_instructions_fn(instructions=instructions_batch, outputs=outputs_batch)

        # the new padding tokens are also non-completion tokens
        instruction_toks_len -= inputs["attention_mask"].sum(dim=1) + inputs["input_ids"].shape[-1]

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0 # loss should not be computed for last token position

        for b in range(inputs["input_ids"].shape[0]):
            loss_mask[b, :instruction_toks_len[b]] = 0

        # # also mask out all tokens before the eoi token region
        # for b in range(inputs["input_ids"].shape[0]):
        #     for i in range(inputs["input_ids"].shape[1]):
        #         # print(inputs["input_ids"][b, i:i+eoi_toks.shape[0]])
        #         # print(eoi_toks)

        #         if torch.all(inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks):
        #             loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
        #             break

        #         # normally the above condition works. but the tokenization instruction tokens in Llama2 is not clean, and so we need this hack
        #         if eoi_toks.shape[0] == 6 and (inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks).sum().item() >= eoi_toks.shape[0] - 2:
        #             loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
        #             break

        inputs['loss_mask'] = loss_mask

        yield inputs

def batch_iterator_text(repo_id, tokenizer, batch_size, max_length, split='train'):
    """Yields (padded) batches from a text dataset."""
    dataset = load_dataset(repo_id, split=split, streaming=True, trust_remote_code=True)

    it_dataset = iter(dataset)
    while True:
        batch = list(itertools.islice(it_dataset, batch_size))
        if not batch:
            break
        inputs = tokenizer([b['text'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0 # loss should not be computed for last token position

        yield inputs, loss_mask

def get_dataset_iterator(tokenizer: AutoTokenizer, task_config: EvaluationTaskConfig, batch_size: int):

    split = "train" if task_config.num_dataset_samples is None else f"train[:{task_config.num_dataset_samples}]"

    if task_config.use_padding and task_config.chat_format:
        chat_template_name = tokenizer.name_or_path.split("/")[-1]
        print(f"Using chat_template: {chat_template_name}")

        # we have to define a tokenization to pass to the iterator that will build the batches
        def tokenize_instructions_fn(instructions: List[str], outputs: List[str]):
            if task_config.do_generations:
                outputs = [""] * len(instructions)
            formatted_prompts = format_prompts(tokenizer, instructions, outputs, template_name=chat_template_name, remove_bos_token=False, max_output_len=task_config.max_seq_len)
            return tokenizer(
                formatted_prompts,
                padding=True,
                truncation=True,
                max_length=task_config.max_seq_len,
                return_tensors="pt",
            ) 

        return batch_iterator_chat(
            repo_id=task_config.dataset_path,
            tokenize_instructions_fn=tokenize_instructions_fn,
            batch_size=batch_size,
            eoi_toks=tokenizer.encode("<|end_of_instruction|>"),
            split=split,
        )
    elif task_config.use_padding and not task_config.chat_format:
        return batch_iterator_text(
            repo_id=task_config.dataset_path,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_len=task_config.max_seq_len,
            split=split,
        )
    else:
        # Load and tokenize dataset
        split = "train" if task_config.num_dataset_samples is None else f"train[:{task_config.num_dataset_samples}]"
        dataset = load_dataset(task_config.dataset_path, split=split, trust_remote_code=True)

        if task_config.chat_format:
            tokenized_dataset = chunk_and_tokenize_chat(
                dataset,
                tokenizer,
                instruction_key="instruction",
                output_key="output",
                max_seq_len=task_config.max_seq_len,
                remove_bos_token=False, # by default we don't remove the bos token from the prompts
                batch_size=batch_size,
            )
        else:
            tokenized_dataset = chunk_and_tokenize(
                dataset,
                tokenizer,
                text_key='text',
                max_seq_len=task_config.max_seq_len,
            )

        return DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

