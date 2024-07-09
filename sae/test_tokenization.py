# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

# %%
import sys
sys.path.append("..")

# %%
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from data import chunk_and_tokenize, chunk_and_tokenize_chat_fast

# %%

dataset = load_dataset("teknium/openhermes", streaming=False, split="train")

dataset = dataset.filter(lambda x: x['input'] == "")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# %%

def get_token_length(example):
    messages = [
        {"role": "system", "content": example['instruction']},
        {"role": "user", "content": example['output']}
    ]
    token_length = len(tokenizer.apply_chat_template(messages, tokenize=True))
    return token_length

dataset = dataset.map(lambda x: {'token_length': get_token_length(x)})

dataset = dataset.sort("token_length", reverse=True)

# %%

tokenized = chunk_and_tokenize_chat_fast(
    dataset,
    tokenizer,
    max_seq_len=1024,
    batch_size=32,
    num_proc=1,
)

# %%

tokenizer.name_or_path

# %%

tokenized = chunk_and_tokenize(
    dataset,
    tokenizer,
    max_seq_len=1024,
    num_proc=1,
    text_key='instruction',
)

# %%

tokenized = chunk_and_tokenize_chat_fast(
    dataset,
    tokenizer,
    max_seq_len=1024,
    num_proc=1,
    instruction_key='instruction',
    output_key='output',
)

# %%

tokenizer.eos_token

# %%

# Create DataLoader
batch_size = 32
dataloader = DataLoader(tokenized, batch_size=batch_size)

# %%

# Example usage
for batch in dataloader:
    # Process your batch here
    # Each element in the batch now contains multiple packed conversations
    # separated by the <|endoftext|> token
    print(batch.shape)  # Should be [batch_size, max_length]
    # Use batch for further processing
# %%
