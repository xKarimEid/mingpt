from gpt_network import GPT
from config import GPTConfig
from trainer import Trainer 


model = GPT(GPTConfig())

trainer = Trainer(model)
#trainer.train_model()