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

    def __init__(self, config, scaling):
        super().__init__()

        scaling = torch.tensor(scaling)
        if scaling.shape != (2,):  # not the same mean&std for all tensor channels
            scaling = scaling.unsqueeze(-1).unsqueeze(-1)
        self.mean = scaling[0]
        self.std = scaling[1]

        self.model = _get_nnaudio_transform(config)

    def forward(self, x: torch.Tensor):
        b = x.size(0)
        x = x.flatten(0, 1)
        x = self.model(x)
        x = x.unflatten(0, (b, 3))
        x = x/self.std - self.mean
    
        return x
