"""
Implement transformer again by blocks
"""

import torch
import torch.nn.functional as F
from torch import nn

from dataclasses import dataclass 

import math 

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        # Creating keys, values, queries in one matrix
        self.c_attn = nn.Linear(config.n_embed, config.n_embed*3)
        # output projections
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.register_buffer('bias', 
            torch.tril(torch.ones(config.context_size, config.context_size))
                        .view(1, 1, config.context_size, config.context_size))
        
        self.n_heads = config.n_heads 
        self.head_size = config.head_size
        self.n_embed = config.n_embed

    def forward(self, x):
        
        B, T, C = x.shape
        kqv = self.c_attn(x)
        k, q, v = kqv.split(self.n_embed, dim=-1)
        
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        wei = k @ q.transpose(-2, -1) * (1/math.sqrt(k.size(-1)))
        wei = wei.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        out = wei @ v # (B, nh, T, T) (B, nh, T, hs) - (B, nh, T, hs) 
        
        #out = out.view(B, T, config.n_embed)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)

        return out 


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embed*4, config.n_embed)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

@dataclass
class GPTConfig:
    context_size: int = 1024
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    n_embed: int =  768
    head_size: int = int(n_embed/n_heads)


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed), 
            wpe = nn.Embedding(config.context_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]), 
            ln_f = nn.LayerNorm(config.n_embed), 
            
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)
        self.context_size = config.context_size
        
    
    def forward(self, idx):
        
        B, T = idx.size()
        assert T <= self.context_size

        te = self.transformer.wte(idx) 
        pe = self.transformer.wpe(torch.arange(T, dtype=torch.long, device = idx.device))
        x = te + pe
        
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
    
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embed are determined from model_type
        config_args = {
            'gpt2':         dict(n_layers=12, n_heads=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layers=24, n_heads=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layers=36, n_heads=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layers=48, n_heads=25, n_embed=1600), # 1558M params
        }[model_type]
        
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['context_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


model = GPT.from_pretrained('gpt2')
model.eval()


import tiktoken


enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")

num_samples = 5

idx = torch.tensor(tokens, dtype=torch.long)
idx = idx.unsqueeze(0).repeat(num_samples, 1)

torch.manual_seed(42)

while idx.size(1) < 30:
    with torch.no_grad():
        logits = model(idx)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_idx, -1, ix)
        idx = torch.cat((idx, xcol), dim=-1)

tokens = idx.tolist()
for token in tokens:
    print(enc.decode(token))

