import os

from tqdm.auto import tqdm
from omegaconf import OmegaConf
from os.path import abspath
from glob import glob
import numpy as np
from torch.utils.data import DataLoader


from .datasets import TransformDataset
from .transforms import get_nnaudio_transform
from ..utils import get_logger


_logger = get_logger()



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


def _preprocess(yml_path, dataset_path, output_path):
    # read preprocessing pipeline
    config = _read_config(yml_path)
    
    # read stacking integer
    stacking = config.stacking

    # read normalization vectors
    mean, std = map(np.array, config.scaling)

    # create spectrogram tansform
    transform = get_nnaudio_transform(config)

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
            spec = (spec[i] - mean) / std
            # stack along this axis: `stacking`
            spec = np.concatenate([spec[_i] for _i in range(3)], axis=stacking)
            # save
            np.save(name, spec)
    
    _logger.info('The data generation process is finished')


def stft(in_path, out_path, ):
    _preprocess(os.path.join(_CONFIGS, 'stft.yaml'), in_path, out_path, )


def mel(in_path, out_path, ):
    _preprocess(os.path.join(_CONFIGS, 'mel.yaml'), in_path, out_path, )