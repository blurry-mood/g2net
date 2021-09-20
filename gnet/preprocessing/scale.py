from torch import nn
import torch


class Scale(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.mean = torch.tensor(cfg.scaling.mean)
        self.std = torch.tensor(cfg.scaling.std)

    def forward(self, x):
        x = (x  - self.mean)/self.std
        return x