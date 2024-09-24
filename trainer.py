

import math
from config import TrainConf
from prepare_data import DataLoader
import time
import torch

class Trainer:
    """Trains a model using """

    def __init__(self, model, train_conf = TrainConf(), dataloader=DataLoader()):
        self.model = model
        self.dataloader = dataloader
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
        # Crop fp precision for matrix multiplies to use tf32 instead of fp32
        # This is available on A series gpus
        #torch.set_float32_matmul_precision('high')
        
        for step in range(self.config.max_steps):
            t1 = time.time()
            # Get the next batch
            xb, yb = self.dataloader.next_batch()
            xb, yb = xb.to('cuda'), yb.to('cuda')
            
            lr = self._get_lr(step)
            self.optim.zero_grad()
            loss_accum = 0.0
            
            logits, loss = self.model(xb, yb)
            
            loss.backward()
            #Clips the gradient
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            
            torch.cuda.synchronize()
            
            t2 = time.time()

            dt = (t2 - t1)*1000

            tokens = self.config.B * self.config.T * self.config.num_processes
            tokens_per_sec = tokens/(t2-t1)

            print(f"step: {step}, loss: {loss.item():.4f}, 
                  norm: {norm}, lr: {lr:.4f}, dt: {dt:.0f}ms, 
                  tok/s: {tokens_per_sec:.0f}")

            # with torch.autocast(device_type='cuda', dtype=bfloat16):
            #    logits, loss = models(xb, yb)
