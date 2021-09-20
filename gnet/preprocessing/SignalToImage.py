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
            config = dict(cfg[i])
            # config['stride'] = tuple(config['stride'])
            mods.append(nn.Conv2d(**config))
            mods.append(nn.MaxPool2d(kernel_size=5, stride=(1, 2), padding=2))
            mods.append(nn.BatchNorm2d(config['out_channels']))
            mods.append(nn.ELU())

        self.model = nn.Sequential(*mods)

    def forward(self, x):
        x = (x  - self.mean)/self.std
        x.unsqueeze_(1)
        x = self.model(x)
        x = x.permute(0, 2, 1, 3)
        return x