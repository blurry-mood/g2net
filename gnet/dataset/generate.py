import os
import sys
import logging

from tqdm.auto import tqdm
from nnAudio.Spectrogram import STFT, MelSpectrogram
from omegaconf import OmegaConf
from os.path import abspath
from glob import glob
import numpy as np
from torch.utils.data import DataLoader

if __name__=='__main__':
    from datasets import TransformDataset
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s:%(message)s')
    _logger = logging.getLogger()
else:
    from .datasets import TransformDataset
    from ..utils import get_logger
    _logger = get_logger()



_TRANSFORMS = {'stft': STFT, 'mel': MelSpectrogram}
_CONFIGS = os.path.join(os.path.split(__file__)[0], 'config')


def _validate_config(config):
    assert config.transform is not None
    assert config.stacking is not None
    assert config.scaling is not None
    assert config.dataloader is not None

    cfg = config.transform
    assert cfg.name is not None


def _read_config(yml_path):
    config = OmegaConf.load(yml_path)
    OmegaConf.set_readonly(config, True)
    _validate_config(config)
    return config


def _get_transform(config):
    transform = config.transform.name
    transform = _TRANSFORMS[transform]
    args = dict(config.transform)
    args.pop('name', None)
    transform = transform(**args)
    return transform


def _preprocess(yml_path, dataset_path, output_path):
    # read preprocessing pipeline
    config = _read_config(yml_path)
    # read stacking integer
    stacking = config.stacking
    # read normalization vectors
    mean, std = map(np.array, config.scaling)

    # create spectrogram tansform
    transform = _get_transform(config)

    # fetch the training data files
    dataset_path, output_path = map(abspath, [dataset_path, output_path])

    _logger.info(f'Fetching the files from: {dataset_path}')

    npys = glob(os.path.join(dataset_path, '**', '*.npy'), recursive=True)
    # logging messages
    _logger.info(f'The preprocessed files will be saved under: {output_path}')
    _logger.info(f'{len(npys)} files have been found')

    # create dataset & dataloader
    dataset = TransformDataset(npys, transform)
    dataloader = DataLoader(dataset, **dict(config.dataloader))

    for inds, specs in tqdm(dataloader):
        inds = inds.tolist()
        specs = specs.numpy()

        for i in range(specs.shape[0]):
            # output path
            name = os.path.join(output_path, os.path.split(npys[inds[i]])[-1])
            # rescale
            spec = (specs[i] - mean) / std
            # save
            np.save(name, spec)


def stft(in_path, out_path, ):
    _preprocess(os.path.join(_CONFIGS, 'stft.yaml'), in_path, out_path, )


def mel(in_path, out_path, ):
    _preprocess(os.path.join(_CONFIGS, 'mel.yaml'), in_path, out_path, )