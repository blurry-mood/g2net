from glob import glob
import os
from omegaconf import OmegaConf
from torch import nn

from ..utils import get_logger
from .spec_transform import SpecTransform
from .concatenator import HStack, FFTStack

_logger = get_logger()
_HERE = os.path.split(__file__)[0]


class Preprocessor(nn.Module):

    def __init__(self, config_name):
        super().__init__()

        config = glob(os.path.join(
            _HERE, '**', f'{config_name}.yaml'), recursive=True)
        if config == []:
            _logger.error(
                f'Cannot find the specified config file! {config_name}.yaml do not exist in r {_HERE}/config')
            raise ValueError()
        config = OmegaConf.load(config[0])

        self.spec_transform = SpecTransform(config.transform, config.scaling)
        if self.spec_transform.m_fft:
            self.concatenator = FFTStack()
        elif config.stacking>=0:
            self.concatenator = HStack(config.stacking)
        self.stack = self.spec_transform.m_fft or (config.stacking>=0)

    def forward(self, x):
        x = self.spec_transform(x)
        if self.stack:
            x = self.concatenator(x)
        return x
