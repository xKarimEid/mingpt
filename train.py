import torch 
import time

from prepare_data import DataLoader
from gpt_network import GPT, GPTConfig

torch.manual_seed(13424)
if torch.cuda.is_available:
    torch.cuda.manual_seed(13424)

B, T = 8, 1024

train_loader = DataLoader(B, T)

torch.set_float32_matmul_precision('high')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

model = GPT(GPTConfig())
model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr = 3e-4 )
for i in range(50):
    t0 = time.time()
    xb, yb = train_loader.next_batch()
    xb, yb = xb.to(device), yb.to(device)
    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(xb, yb)
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    if device == 'cuda':
        torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1-t0)*1000

    tokens_per_sec = train_loader.B * train_loader.T / (t1-t0)
    print(f"step {i} with loss: {loss.item():.4f}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
