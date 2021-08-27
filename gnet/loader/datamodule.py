from typing import Optional


from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from glob import glob
import os
import pandas as pd

from omegaconf import OmegaConf

from .dataset import RawDataset
from ..utils import get_logger

_logger = get_logger()
_HERE = os.path.split(__file__)[0]


class DataModule(pl.LightningDataModule):

    def __init__(self, data_path, config_name, ):
        super().__init__()
        config = glob(os.path.join(
            _HERE, '**', f'{config_name}.yaml'), recursive=True)
        if config == []:
            _logger.error(
                f'Cannot find the specified config file! {config_name}.yaml do not exist in the config folder')
            raise ValueError()
        config = OmegaConf(config[0])

        # attrs
        self.split = config.split
        self.dataloader = dict(config.dataloader)
        self.data_path = data_path

    def prepare_data(self):
        paths = glob(os.path.join(self.data_path,
                                  '**', '*.npy'), recursive=True)
        csv = glob(os.path.join(self.data_path, '*.csv'))
        if csv == []:
            _logger.error(
                f'Cannot find the CSV file of labels! No csv file exists within {self.data_path}')

        csv = csv[0]
        _logger.info(f'{len(paths)} data sample has been found')
        _logger.info(f'The csv file is found: {csv}')

        df = pd.read_csv(csv)
        labels = []
        dictt = {row[0]: row[1] for row in df.values}
        for path in tqdm(paths, desc='Map inputs to labels', leave=False):
            name = os.path.split(path)[-1].split('.')[0]
            labels.append(dictt.pop(name))

        self.dataset = RawDataset(paths, labels)
        _logger.info("The dataset is created")

    def setup(self, stage: Optional[str]):
        if stage == 'fit' or stage is None:
            n = len(self.dataset)

            val, test = map(lambda x: int(
                x*n), [self.split.val, self.split.test])
            train = n - (test + val)

            self.train, self.val, self.test = random_split(
                self.dataset, [train, val, test])

            _logger.info(
                f"The dataset split is performed: train={train}, val={val}, test={test}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, **self.dataloader)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, **self.dataloader)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, **self.dataloader)
