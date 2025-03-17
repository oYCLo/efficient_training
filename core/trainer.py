from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class MixedPrecisionTrainer:
    def __init__(self):
        self.scaler = GradScaler()
        
    def training_step(self, model, data, optimizer):
        with autocast():
            loss = model(data)
            
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
class GradientAccumulator:
    def __init__(self, accumulation_steps=4):
        self.accumulation_steps = accumulation_steps
        
    def train_step(self, model, data, optimizer, step):
        loss = model(data) / self.accumulation_steps
        loss.backward()
        
        if (step + 1) % self.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


class DistributedTrainer:
    def __init__(self):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
    def setup_model(self, model):
        return DistributedDataParallel(model)