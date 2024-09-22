
import os 
import math
from config import TrainConf
from torch.distributed import init_process_group, destroy_process_group
import torch 


class GPUTrain:
    def __init__(self, model):
        self.model = model
        self.config = TrainConf()
        self.optim = model.module.configure_optimizers(self.config.weight_decay, 
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
        for i in range(self.config.max_steps):
            loss_accum = 0.0
            self.optim.zero_grad()



class DDPTrain:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    print(f"I am GPU with ddp_rank: {ddp_rank}, ddp_local_rank: {ddp_local_rank}, ddp_world_size: {ddp_world_size}")
    print("Bye")
