"""
Code for loading gpt2 weights (124M parameters) and sampling from it
"""


from transformers import GPT2LMHeadModel
from transformers import pipeline, set_seed


set_seed(42)

gen = pipeline('text-generation', model='gpt2')

gen("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
