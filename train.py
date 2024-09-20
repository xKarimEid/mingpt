import torch 
import time
import math 

from prepare_data import DataLoader
from gpt_network import GPT, GPTConfig
from torch.distributed import init_process_group, destroy_process_group
import os

torch.manual_seed(13424)
if torch.cuda.is_available:
    torch.cuda.manual_seed(13424)

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    print(f"I am GPU {ddp_rank}")
    print("Bye")

print("finished!")
import sys; sys.exit(0)

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 16 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
train_loader = DataLoader(B=B, T=T)

torch.set_float32_matmul_precision('high')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps = 50

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return 0.1 * max_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

#optim = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas=(0.9, 0.95), eps=1e-8)
# optimize!
optim = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for i in range(max_steps):
    t0 = time.time()
    
    loss_accum = 0.0
    optim.zero_grad()
    for micro_step in range(grad_accum_steps):
        xb, yb = train_loader.next_batch()
        xb, yb = xb.to(device), yb.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(xb, yb)

        loss = loss/grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    # If norm is increasing, then training is destabilizing
    # Clipping the gradient by the norm
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Getting the lr based on our cosine warmup function
    lr = get_lr(i)
    # Setting the lr in our optimizer
    for param_group in optim.param_groups:
        param_group['lr'] = lr
    # Updating the weights
    optim.step()
    
    if device == 'cuda':
        torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1-t0)*1000

    tokens_per_sec = train_loader.B * train_loader.T * grad_accum_steps / (t1-t0)

    print(f"step {i} with loss: {loss_accum.item():.4f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
