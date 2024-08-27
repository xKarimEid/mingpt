
import os
from gpt_network import GPT, GPTConfig
from generate import generate_tokens
import tiktoken


data_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(data_path, 'r') as f:
    data = f.readlines()

enc = tiktoken.get_encoding('gpt2')


# Implement loss calculation

# Implement validation

# Setup train loop

