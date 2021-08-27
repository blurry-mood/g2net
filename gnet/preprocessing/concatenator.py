from typing import List
import torch
from torch import nn


class HStack(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.cat([x[:, i:i+1] for i in range(3)], dim=2 + self.dim)
        x = torch.tile(x, dims=[1, 3, 1, 1])
        return x


class FFTStack(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, xx: List[torch.Tensor]):
        for i, x in enumerate(xx): # len(xx)==3
            x = torch.cat([x[:, i:i+1] for i in range(3)], dim=2 + self.dim)
            xx[i] = x
        x = torch.cat(xx, dim=1)
        return x
