"""
Implement gpt2
"""

import torch
import torch.nn.functional as F
from torch import nn

from dataclasses import dataclass 

import math
import inspect

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        # Creating keys, values, queries in one matrix
        self.c_attn = nn.Linear(config.n_embed, config.n_embed*3)
        # output projections
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.INIT_SCALE = 1
        #self.register_buffer('bias', 
        #    torch.tril(torch.ones(config.context_size, config.context_size))
        #                .view(1, 1, config.context_size, config.context_size))

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

        #wei = k @ q.transpose(-2, -1) * (1/math.sqrt(k.size(-1)))
        #wei = wei.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        #wei = F.softmax(wei, dim=-1)
        #out = wei @ v # (B, nh, T, T) (B, nh, T, hs) - (B, nh, T, hs) 

        out = F.scaled_dot_product_attention(k, q, v, is_causal=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.c_proj(out)

        return out 

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embed*4, config.n_embed)
        self.INIT_SCALE = 1
    
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
        self.config = config
        self.context_size = config.context_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed), 
            wpe = nn.Embedding(config.context_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]), 
            ln_f = nn.LayerNorm(config.n_embed), 
            
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)
        
        # Weight sharing parameter
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):    
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'INIT_SCALE'):
                std *= (2*self.config.n_layers)**-0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.context_size

        # Create positions for position embeddings
        pos = torch.arange(T, dtype=torch.long, device = idx.device)
        pe = self.transformer.wpe(pos)
        te = self.transformer.wte(idx) 
        # Encode positions and tokens together
        x = te + pe
        
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(B*T, 50257)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
