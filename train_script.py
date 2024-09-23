from gpt_network import GPT
from config import GPTConfig
from trainer import Trainer 


device='cuda'

model = GPT(GPTConfig())

model.to(device)

trainer = Trainer(model)

#trainer.train_model()