import torch
import torch.nn as nn

class QuantizedInference:
    def __init__(self, model, dtype=torch.qint8):
        
        self.model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=dtype
        )
    
    def inference(self, x):
        with torch.no_grad():
            return self.model(x)
        
class BatchInference:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        
    @torch.no_grad()
    def predict(self, data_loader):
        results = []
        for batch in data_loader:
            output = self.model(batch)
            results.append(output)
        return torch.cat(results, dim=0)