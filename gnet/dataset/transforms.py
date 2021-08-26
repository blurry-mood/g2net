from omegaconf import OmegaConf
from nnAudio.Spectrogram import STFT, MelSpectrogram


_TRANSFORMS = {'stft': STFT, 'mel': MelSpectrogram}

def get_nnaudio_transform(config):
    transform = config.transform.name
    transform = _TRANSFORMS[transform]
    args = dict(config.transform)
    args.pop('name', None)
    transform = transform(**args)
    return transform