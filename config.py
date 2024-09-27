"""
Configuration settings for training on multi GPUs
"""

import os
from dataclasses import dataclass
from torch.distributed import init_process_group
import torch 


@dataclass
class GPTConfig:
    context_size: int = 1024
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 12
    n_embed: int =  768
    head_size: int = int(n_embed/n_heads)


@dataclass
class TrainConf:
    max_lr: float = 6e-4
    min_lr: float = 0.1 * max_lr
    warmup_steps: int = 10
    max_steps: int = 50
    weight_decay: float = 0.1
    learning_rate: float = 6e-4
    total_batch_size: int = 524288
    B: int = 16
    T: int = 1024
    grad_accum_steps: int = total_batch_size // (B*T)
    device: str = 'cuda'

    assert torch.cuda.is_available() == True
    assert total_batch_size % (B*T) == 0 , "total_batch_size has to be divisible by B*T"
    
    # DDP Configuration settings
    ddp: bool = int(os.environ.get('RANK', -1)) != -1
    ddp_rank: int = 0
    ddp_local_rank: int = 0
    ddp_world_size: int = 1
    master_process: bool = True

    def init_ddp(self):
        """
        Initialize DDP settings if DDP is detected
        If DDP is detected gradient accumulation steps decreases
        due to the extra GPUs. 
        """

        if self.ddp:
            # Sets up communication between the GPUs 
            # and initializes the environment
            init_process_group(backend='nccl')
            self.ddp_rank = int(os.environ['RANK'])
            # Local to all gpus on same node
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            
            # Update grad_accum_steps now that we have more than 1 gpu 
            self.grad_accum_steps = int(self.total_batch_size // (self.B*self.T*self.ddp_world_size))
            assert self.grad_accum_steps % (self.B*self.T*self.ddp_world_size) == 0

            # Master process is used for admin tasks
            self.master_process = self.ddp_rank == 0

            # Just for dev purposes
            if self.master_process:
                print(f"Total desired batch size: {self.total_batch_size}")
                print(f"Gradient accumulation steps: {self.grad_accum_steps}")
