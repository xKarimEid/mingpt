"""
Initialize model, move to gpu, compile and train
"""

from gpt_network import GPT
from config import GPTConfig
from trainer import Trainer 
import torch

device='cuda'

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

trainer = Trainer(model)
trainer.train_model()

# Aim for 185k tokens / sec