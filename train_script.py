"""
Initialize model, move to gpu, compile and train
"""

from gpt_network import GPT
from config import GPTConfig, TrainConf
from trainer import Trainer 
import torch

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

# Init model conf and train conf object

model_conf = GPTConfig()
train_conf = TrainConf()
# Init DDP run if available
train_conf.init_ddp()

# Init model using prev conf
model = GPT(model_conf)
# Move model to device
model.to(train_conf.device)
# Compile model for faster run times
model = torch.compile(model)


# Creat Trainer instance
trainer = Trainer(model, train_conf=train_conf)
# Train model
trainer.train_model()

# Aim for 185k tokens / sec, currently getting 175k tokens/sec