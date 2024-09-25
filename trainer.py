"""
Script for training the LLM similar to how GPT2 and GPT3 
were trained.
"""

import math
from config import TrainConf
from prepare_data import DataLoader
import time
import torch


class Trainer:
    """Trains a GPT based LLM using gpt2 and gpt3 references"""

    def __init__(self, model, train_conf = TrainConf(), dataloader=DataLoader()):
        self.model = model
        self.dataloader = dataloader
        self.config = train_conf
        self.optim = model.configure_optimizers(self.config)

    def _get_lr(self, it):
        """Cosine learning rate decay with a warmup"""

        if it < self.config.warmup_steps:
            return self.config.max_lr * (it + 1) / self.config.warmup_steps
        if it > self.config.max_steps:
            return 0.1 * self.config.max_lr
        
        decay_ratio = (it - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.max_lr - self.config.min_lr)
    
    def train_model(self):
        # Crop fp precision for matrix multiplication to use tf32 instead of fp32
        # This is available on A series gpus
        torch.set_float32_matmul_precision('high')
        
        for step in range(self.config.max_steps):
            t1 = time.time()
            loss_accum = 0.0
            self.optim.zero_grad()
            # Implement gradient accumulation to achieve 0.5M batch size
            for micro_step in range(self.config.grad_accum_steps):
                # Get the next batch
                xb, yb = self.dataloader.next_batch()
                xb, yb = xb.to('cuda'), yb.to('cuda')

                # Forward the batch using mixed precision
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = self.model(xb, yb)
                # Scale the loss down because we are going to 
                # add the gradients everytime we calculate loss.backward()
                # So the loss has to be scaled down before loss.backward() is
                # called. 
                loss = loss/self.config.grad_accum_steps
                loss_accum += loss.detach()
                # Calculate the gradients
                loss.backward()

            # Calculate new learning rate
            lr = self._get_lr(step)
            # Set the new learning rate
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr

            # Clips the gradients
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # Update parameters
            self.optim.step()
            # Synchronize and time the step            
            torch.cuda.synchronize()
            t2 = time.time()
            dt = (t2 - t1)*1000
            # Calculate training speed
            tokens = self.config.B * self.config.T * self.config.grad_accum_steps * self.config.num_processes
            tokens_per_sec = tokens/(t2-t1)

            print(f"step: {step}, loss: {loss_accum:.4f}, norm: {norm}, lr: {lr:.4f}, dt: {dt:.0f}ms, tok/s: {tokens_per_sec:.0f}")