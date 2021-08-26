from logging import log
import os

from tqdm.auto import tqdm
from omegaconf import OmegaConf
from os.path import abspath
from glob import glob
import numpy as np
from torch.utils.data import DataLoader


from datasets import TransformDataset
from transforms import get_nnaudio_transform
from logging import DEBUG, INFO, getLogger, Formatter, StreamHandler

__all__ = ['get_logger']

def get_logger():
    logger = getLogger('G2Net')
    logger.setLevel(DEBUG)

    # formatter

    # stream handler
    if not logger.hasHandlers():
        fmr = _ColoredFormatter('%(name)s: %(filename)s:%(lineno)s - %(levelname)s:  %(message)s')
        ch = StreamHandler()
        ch.setLevel(DEBUG)
        ch.setFormatter(fmr)

        logger.addHandler(ch)
    
    return logger


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
#The background is set with 40 plus the number of the color, and the foreground with 30

#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[;%dm"
BOLD_COLOR_SEQ = "\033[1;%dm"

_COLORS = {
    'WARNING': YELLOW,
    'INFO': GREEN,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}

class _ColoredFormatter(Formatter):
    def __init__(self, msg, use_color = True):
        Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in _COLORS:
            levelname_color = COLOR_SEQ % (30 + _COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color

        # name
        name = record.name
        if self.use_color:
            name_color = BOLD_COLOR_SEQ % (30 + RED) + name + RESET_SEQ
            record.name = name_color
        return Formatter.format(self, record)

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
    log_size = True

    for inds, specs in tqdm(dataloader):
        inds = inds.tolist()
        specs = specs.numpy()

        for i in range(specs.shape[0]):
            # output path
            name = os.path.join(output_path, os.path.split(npys[inds[i]])[-1])
            # normalize
            spec = (specs[i] - mean) / std
            # stack along this axis: `stacking`
            spec = np.concatenate([spec[_i][None] for _i in range(3)], axis=stacking)

            if log_size:
                log_size = False
                _logger.info(f'The old shape: {specs[i].shape}, the new input shape: {spec.shape}')

            # save
            np.save(name, spec)

    _logger.info('The data generation process is finished')


def stft(in_path, out_path, ):
    _preprocess(os.path.join(_CONFIGS, 'stft.yaml'), in_path, out_path, )


def mel(in_path, out_path, ):
    _preprocess(os.path.join(_CONFIGS, 'mel.yaml'), in_path, out_path, )


stft('data', 'data2')