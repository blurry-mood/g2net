import os
import logging

from tqdm.auto import tqdm
from nnAudio.Spectrogram import STFT
from omegaconf import OmegaConf
from os.path import abspath
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger()

_TRANSFORMS = {'stft': STFT}
_CONFIGS = os.path.join(os.path.split(__file__)[0], 'config')

class _Dataset(Dataset):

    def __init__(self, paths) -> None:
        super().__init__()
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return i, np.load(self.paths[i]).astype('float32')


def _collate(batch):
    paths, ts = zip(*batch)
    return torch.tensor(paths), torch.cat(list(map(torch.from_numpy, ts)))


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


def _preprocess(yml_path, dataset_path, output_path, device):
    # read preprocessing pipeline
    config = _read_config(yml_path)
    # read stacking integer
    stacking = config.stacking
    # read normalization vectors
    mean, std = map(np.array, config.scaling)

    # create spectrogram tansform 
    tranform = _get_transform(config).to(device)
    
    # fetch the training data files
    dataset_path, output_path = map(abspath, [dataset_path, output_path])
    
    _logger.info(f'fetching the files from: {dataset_path}')

    npys = glob(os.path.join(dataset_path,'**','*.npy'), recursive=True)
    # logging messages
    _logger.info(f'The preprocessed files will be saved under: {output_path}')
    _logger.info(f'{len(npys)} files have been found')

    # create dataset & dataloader
    dataset = _Dataset(npys)
    dataloader = DataLoader(dataset, collate_fn=_collate,
                            **dict(config.dataloader))

    for paths, ts in tqdm(dataloader):
        # 
        paths = paths.tolist()
        ts = ts.to(device)
        # compute spectrogram
        specs = tranform(ts).cpu().numpy()

        for i in range(specs.shape[0]//3):
            # output path
            name = os.path.join(output_path, os.path.split(npys[paths[i]])[-1])
            # concatenate the 3 detector signal
            list = [specs[i+j:i+j+i] for j in range(3)]
            spec = np.concatenate(list, axis=stacking)
            # rescale
            spec = (spec - mean) /std
            # save
            np.save(name, spec)

def stft(in_path, out_path, device):
    _preprocess(os.path.join(_CONFIGS, 'stft.yaml'), in_path, out_path, device)


def mel(in_path, out_path, device):
    _preprocess(os.path.join(_CONFIGS, 'mel.yaml'), in_path, out_path, device)