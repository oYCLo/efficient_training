from torch.utils.checkpoint import checkpoint
import torch.nn as nn
class MemoryEfficientModel(nn.Module):
    def forward(self, x):
        return checkpoint(self.heavy_computation, x)
    
class DynamicBatchSizer:
    def __init__(self, initial_batch_size, max_memory_usage):
        self.batch_size = initial_batch_size
        self.max_memory = max_memory_usage
        
    def adjust_batch_size(self, current_memory_usage):
        if current_memory_usage > self.max_memory:
            self.batch_size = int(self.batch_size * 0.8)
        return self.batch_size