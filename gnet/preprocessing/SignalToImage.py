from torch import nn
import torch


class SignalToImage(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.mean = torch.tensor(cfg.scaling.mean)
        self.std = torch.tensor(cfg.scaling.std)

        cfg = cfg.convnet
        n = len(cfg)

        mods = []
        for i in range(n):
            cfg[i].stride = tuple(cfg[i].stride)
            mods.append(nn.Conv2d(**dict(cfg[i])))
            mods.append(nn.SiLU())

        self.model = nn.Sequential(*mods)

    def forward(self, x):
        x = x / self.std - self.mean
        x.unsqueeze_(1)
        x = self.model(x)
        x = torch.permute(x, dims=(0, 2, 1, 3))
        return x