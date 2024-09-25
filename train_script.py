"""
Initialize model, move to gpu, compile and train
"""

from gpt_network import GPT
from config import GPTConfig, TrainConf
from trainer import Trainer 
import torch

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

# Init model conf object
gpt_conf = GPTConfig()
# Init model using prev conf
model = GPT(gpt_conf)
# Move model to device
model.to(gpt_conf.device)
# Compile model for faster run times
model = torch.compile(model)

# Create train config and init ddp if available
train_conf = TrainConf()
train_conf.init_ddp()

# Creat Trainer instance
trainer = Trainer(model, train_conf=train_conf)
# Train model
trainer.train_model()

# Aim for 185k tokens / sec, currently getting 175k tokens/sec