from typing import List
from nnAudio.Spectrogram import STFT, MelSpectrogram
import torch
from omegaconf import ListConfig

_TRANSFORMS = {'stft': STFT, 'mel': MelSpectrogram}


def _get_nnaudio_transform(config):
    transform = config.name
    transform = _TRANSFORMS[transform]
    args = dict(config)
    args.pop('name', None)

    win_lengths = args.pop('win_length')

    # if 3 nfft values are supplied, do the following
    if isinstance(win_lengths, ListConfig):
        transforms = torch.nn.ModuleList()
        for win_length in win_lengths:
            transforms.append(transform(win_length=win_length, **args))
        return transforms

    transform = transform(win_length=win_lengths, **args)

    return transform


def _transforms(modules: torch.nn.ModuleList, mean: torch.Tensor, std: torch.Tensor, x: torch.Tensor):
    xx = []
    for transform in modules:
        xx.append(_transform(transform, mean, std, x))
    return xx


def _transform(module: torch.nn.Module, mean: torch.Tensor, std: torch.Tensor, x: torch.Tensor):
    b = x.size(0)
    x = x.flatten(0, 1)
    x = module(x)
    x = x.unflatten(0, (b, 3))
    x = x/std - mean
    return x


class SpecTransform(torch.nn.Module):

    def __init__(self, config, scaling):
        super().__init__()

        scaling = torch.tensor(scaling)
        if scaling.shape != (2,):  # not the same mean&std for all tensor channels
            scaling = scaling.unsqueeze(-1).unsqueeze(-1)
        self.mean = scaling[0]
        self.std = scaling[1]

        self.mods = _get_nnaudio_transform(config)
        self._win_lengths = isinstance(
            self.mods, torch.nn.ModuleList)

        self.func = _transforms if self._win_lengths else _transform

    @property
    def multi_win_lengths(self):
        return self._win_lengths

    def forward(self, x: torch.Tensor):
        x = self.func(self.mods, self.mean, self.std, x)
        return x
