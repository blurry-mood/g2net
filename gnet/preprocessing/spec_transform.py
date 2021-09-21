from typing import List
from nnAudio.Spectrogram import STFT, MelSpectrogram, CQT2010v2
import torch

_TRANSFORMS = {'stft': STFT, 'mel': MelSpectrogram, 'cqt': CQT2010v2}


def _get_nnaudio_transform(config):
    transform = config.name
    transform = _TRANSFORMS[transform]
    args = dict(config)
    args.pop('name', None)

    transform = transform(**args)

    return transform


class SpecTransform(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = _get_nnaudio_transform(config)

    def forward(self, x: torch.Tensor):
        b, len, _ = x.shape
        x = x.flatten(0, 1)
        x = self.model(x)
        x = x.unflatten(0, (b, len))
        return x
