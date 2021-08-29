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

    n_fft = args.pop('n_fft')

    # if 3 nfft values are supplied, do the following
    if isinstance(n_fft, ListConfig):
        transforms = torch.nn.ModuleList()
        for fft in n_fft:
            transforms.append(transform(n_fft=fft, **args))
        return transforms

    transform = transform(n_fft=n_fft, **args)

    return transform


def _transforms(modules: torch.nn.ModuleList, x: torch.Tensor):
    xx = []
    for transform in modules:
        xx.append(_transform(transform, x))
    return xx


def _transform(module: torch.nn.Module, x: torch.Tensor):
    b = x.size(0)
    x = x.flatten(0, 1)
    x = module(x)
    x = x.unflatten(0, (b, 3))
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
        self._mfft = isinstance(
            self.mods, torch.nn.ModuleList)
        
        self.func = _transforms if self._mfft else _transform

    @property
    def m_fft(self):
        return self._mfft

    def forward(self, x: torch.Tensor):
        x = (x - self.mean)/self.std
        x = self.func(self.mods, x)
        return x
