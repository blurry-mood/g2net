from copy import deepcopy
from typing import List
from nnAudio.Spectrogram import STFT, MelSpectrogram
import torch


_TRANSFORMS = {'stft': STFT, 'mel': MelSpectrogram}


def _get_nnaudio_transform(config):
    transform = config.name
    transform = _TRANSFORMS[transform]
    args = dict(config)
    args.pop('name', None)

    n_fft = args.pop('n_fft')

    # if 3 nfft values are supplied, do the following
    if isinstance(n_fft, List):
        transforms = torch.nn.ModuleList()
        for fft in n_fft:
            transforms.append(transform(n_fft=fft, **args))
        return transforms

    transform = transform(n_fft=n_fft, **args)

    return transform


def _tranforms(modules: torch.nn.ModuleList, mean: torch.Tensor, std: torch.Tensor, x: torch.Tensor):
    xx = []
    for transform in modules:
        xx.append((transform(x) - mean) / std)
    return xx


def _transform(module: torch.nn.Module, mean: torch.Tensor, std: torch.Tensor, x: torch.Tensor):
    return (module(x) - mean)/std


class SpecTransform(torch.nn.Module):

    def __init__(self, config, scaling):
        super().__init__()

        scaling = torch.tensor(scaling)
        if scaling.shape != (2,): # not the same mean&std for all tensor channels
            scaling = scaling.unsqueeze(-1).unsqueeze(-1)
        self.mean = scaling[0]
        self.std = scaling[1]

        self.mods = _get_nnaudio_transform(config)
        self._mfft = isinstance(
            self.mods, torch.nn.Module)
        self.func = _transform if self._mfft else _tranforms
    
    @property
    def m_fft(self):
        return self._mfft

    def forward(self, x):
        return self.func(self.mods, self.mean, self.std, x)
