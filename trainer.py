"""
Script for training the LLM similar to how GPT2 and GPT3 
were trained.
"""

import math
from prepare_data import DataLoader
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import destroy_process_group


class Trainer:
    """Trains a GPT based LLM using gpt2 and gpt3 references"""

    def __init__(self, model, train_conf):
        # Initialize dataloader
        self.train_conf = train_conf
        self.dataloader = DataLoader(self.train_conf)
        self.optim = model.configure_optimizers(train_conf)

        # Wrap model around DDP if training is with DDP
        # DDP is responsible for averaging gradients across ranks
        # and depositing the gradients on all the ranks
        # This is done during/after the backward pass
        if train_conf.ddp:
            self.model = DDP(model, device_ids = [train_conf.ddp_local_rank])
        else:
            self.model = model
        
    def _get_lr(self, it):
        """Cosine learning rate decay with a warmup"""

        # Shortening variable names for better readability
        warmup_steps = self.train_conf.warmup_steps
        max_lr = self.train_conf.max_lr
        max_steps = self.train_conf.max_steps
        min_lr = self.train_conf.min_lr

        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return 0.1 * max_lr
        
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    
    def train_model(self):
        # Crop fp precision for matrix multiplication to use tf32 instead of fp32
        # This is available on A series gpus

        # Shortening variable names for readability
        max_steps = self.train_conf.max_steps
        B = self.train_conf.B
        T = self.train_conf.T
        grad_accum_steps = self.train_conf.grad_accum_steps
        ddp_world_size = self.train_conf.ddp_world_size
        ddp = self.train_conf.ddp
        device = self.train_conf.device

        torch.set_float32_matmul_precision('high')
        
        for step in range(max_steps):
            t1 = time.time()
            loss_accum = 0.0
            self.optim.zero_grad()
            # Implement gradient accumulation to achieve 0.5M batch size
            for micro_step in range(grad_accum_steps):
                # Get the next batch
                xb, yb = self.dataloader.next_batch()
                xb, yb = xb.to(device), yb.to(device)

                # Forward the batch using mixed precision
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = self.model(xb, yb)
                # Scale the loss down because we are going to 
                # add the gradients everytime we calculate loss.backward()
                # So the loss has to be scaled down before loss.backward() is
                # called. 
                loss = loss/grad_accum_steps
                loss_accum += loss.detach()
                # Calculate the gradients
                
                if ddp: 
                    # If its a DDP run, loss.backwards() synchronizes the gradients
                    # But we dont want to do it after every micro step
                    self.model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                
                loss.backward()
            
            if ddp:
                pass
                # If DDP run, we want to see the average loss across
                # All the gpus, 
                #dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)


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
            tokens = B * T * grad_accum_steps * ddp_world_size
            tokens_per_sec = tokens/(t2-t1)

            print(f"step: {step}, loss: {loss_accum:.4f}, norm: {norm:.4f}, lr: {lr:.4f}, dt: {dt:.0f}ms, tok/s: {tokens_per_sec:.0f}")
            print(f"Destroying gpu: {device}")
            
            if ddp:
                destroy_process_group()
            
            import sys; sys.exit(0)