

import math
from config import TrainConf
from prepare_data import DataLoader

class Trainer:
    """Trains a model using """
    def __init__(self, model, train_conf = TrainConf()):
        self.model = model
        self.dataloader = DataLoader()
        self.config = train_conf
        self.optim = model.configure_optimizers(self.config.weight_decay, 
                                                   self.config.learning_rate)

    def _get_lr(self, it):
        if it < self.config.warmup_steps:
            return self.config.max_lr * (it + 1) / self.config.warmup_steps
        if it > self.config.max_steps:
            return 0.1 * self.config.max_lr
        
        decay_ratio = (it - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.max_lr - self.config.min_lr)
    
    def train_model(self):
        
        for step in range(self.config.max_steps):
            # Get the next batch
            xb, yb = self.dataloader.next_batch()
            lr = self._get_lr(step)
            self.optim.zero_grad()

            loss_accum = 0.0
            
            logits, loss = self.model(xb, yb)
            
            loss.backward()

            self.optim.step()

            print(f"loss: {loss.item}, lr: {lr}")
