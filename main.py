"""
Implement transformer again by blocks
"""

import torch
from dataclasses import dataclass

import torch.nn as nn 

n_embed = 50257
block_size = 1024
head_size = 64

class Attention(nn.Module):
    """
    Batch implementation of Casual self attention
    """

    def __init__(self):
        super().__init__()
        self.keys = nn.Linear(n_embed, head_size)
        self.queries = nn.Linear(n_embed, head_size)
        self.values = nn.Linear(n_embed, head_size)
    
    def forward(self, x):
        B, T, C = x.shape
        
        k = self.keys(x) # (B, T, C) (C, head_size) -> (B, T, head_size)
        q = self.queries(x) # (B, T, C) (C, head_size) -> (B, T, head_size)
        v = self.values(x) # (B, T, C) (C, head_size) -> (B, T, head_size)

        wei = k @ q.transpose(-1, -2)

        return wei @ v


class Block(nn.Module):
    def __init__(self):
        pass 

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        pass 


x = torch.randn(size=(3, block_size, n_embed))
print(x.shape)

sa = Attention()

o = sa(x)
print(o.shape)

