from dataclasses import dataclass

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
    
    # For DDP
    num_processes: int = 1
    process_rank: int = 0
    
    assert total_batch_size % (B*T) == 0
