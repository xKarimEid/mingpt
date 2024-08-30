
import torch 

from prepare_data import DataLoader
from gpt_network import GPT, GPTConfig


B, T = 4, 6

train_data = DataLoader(B, T)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

model = GPT(GPTConfig())
model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr = 3e-4 )
for i in range(50):
    xb, yb = train_data.next_batch()
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"step {i} with loss: {loss.item()}")
