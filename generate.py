"""
Generates token from a model
"""

import torch 
import tiktoken
from gpt_network import GPT
import torch.nn.functional as F


def generate_tokens(model):
    
    model.eval()

    device = "cpu"
    if torch.cuda.is_available():
        device = 'cuda'
    
    print(f"using device: {device}")

    #model.to(device)


    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a language model,")

    num_samples = 5

    idx = torch.tensor(tokens, dtype=torch.long)
    idx = idx.unsqueeze(0).repeat(num_samples, 1)
    idx = idx.to(device)

    torch.manual_seed(42)

    while idx.size(1) < 30:
        with torch.no_grad():
            logits, loss = model(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_idx = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_idx, -1, ix)
            idx = torch.cat((idx, xcol), dim=-1)

    tokens = idx.tolist()
    for token in tokens:
        print(enc.decode(token))

if __name__ == '__main__':
    model = GPT.from_pretrained('gpt2')
    generate_tokens(model)

