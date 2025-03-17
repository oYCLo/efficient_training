import torch.profiler
import torch
class PerformanceProfiler:
    def __init__(self):
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=2
            )
        )
    
    def start(self):
        self.profiler.start()
        
    def stop(self):
        self.profiler.stop()
        print(self.profiler.key_averages().table())
        
class MemoryTracker:
    @staticmethod
    def get_memory_stats():
        return {
            "allocated": torch.cuda.memory_allocated(),
            "cached": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated()
        }