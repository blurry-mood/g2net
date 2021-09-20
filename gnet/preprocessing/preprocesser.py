from glob import glob
from gnet.preprocessing.scale import Scale
import os
from omegaconf import OmegaConf
from torch import nn

from ..utils import get_logger
from .spec_transform import SpecTransform
from .SignalToImage import SignalToImage

_logger = get_logger()
_HERE = os.path.split(__file__)[0]


class Preprocessor(nn.Module):

    def __init__(self, config_name):
        super().__init__()

        config = glob(os.path.join(
            _HERE, '**', f'{config_name}.yaml'), recursive=True)
        if config == []:
            _logger.error(
                f'Cannot find the specified config file! {config_name}.yaml do not exist in {_HERE}/config')
            raise ValueError()
        config = OmegaConf.load(config[0])

        # SignalToImage triggered if `convnet` key is present in `config`
        if 'convnet' in dict(config):
            self.transform = SignalToImage(config)
        elif 'transform' in dict(config):
            self.transform = SpecTransform(config.transform, config.scaling)
        elif 'scaling' in dict(config):
            self.transform = Scale(config)

    def forward(self, x):
        x = self.transform(x)
        return x
