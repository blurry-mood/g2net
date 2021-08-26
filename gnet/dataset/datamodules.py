from typing import Optional

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from glob import glob
import os
import pandas as pd

from .datasets import SpecDataset
from ..utils import get_logger

_logger = get_logger()

class DataModule(pl.LightningDataModule):

    def __init__(self, path, config, split, ):
        super().__init__()
        self.path = path
        self.config = dict(config)
        self.split = split

    def prepare_data(self):
        paths = glob(os.path.join(self.path, '**', '*.npy'), recursive=True)
        csv = glob(os.path.join(self.path, '*.csv'))[0]

        _logger.info(f'{len(paths)} data sample has been found')
        _logger.info(f'The csv file is found: {csv}')

        df = pd.read_csv(csv)
        labels = []
        dictt = {row[0]:row[1] for row in df.values}
        for path in tqdm(paths, desc='Map inputs to labels', leave=False):
            name = os.path.split(path)[-1].split('.')[0]
            labels.append(dictt.pop(name))

        self.dataset = SpecDataset(paths, labels)
        _logger.info("The dataset is successfully created")

    def setup(self, stage: Optional[str]):
        if stage == 'fit' or stage is None:
            n = len(self.dataset)
            val, test = map(lambda x: int(
                x*n), [self.split.val, self.split.test])
            train = n - (test + val)
            self.train, self.val, self.test = random_split(
                self.dataset, [train, val, test])
            _logger.info(f"The dataset split is successfully performed: train={train}, val={val}, test={test}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, **self.config)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, **self.config)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, **self.config)
