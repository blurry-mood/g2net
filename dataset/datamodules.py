import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()