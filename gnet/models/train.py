import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb

from ..dataset.datamodules import DataModule
from .litmodel import LitModel
import os

# arguments
parser = argparse.ArgumentParser(description='Train Lightning Module.')
parser.add_argument('--yaml', type=str, required=True,
                    help='YAML config file containing the lit model description.')
parser.add_argument('--data', type=str, required=True,
                    help='Path to training dataset. The training folder should contain train_labels.csv with keys [id, target]. Wildcards are supported.')
args = parser.parse_args()

cfg = os.path.join(os.path.split(__file__)[0], 'config', args.yaml)
data_path = args.data

# model & datamodule
litmodel = LitModel(cfg)
dm = DataModule(data_path, cfg.datamodule, cfg.split)

# wandb logger & lr monitor
wandb.login()
logger = WandbLogger(entity='blurry-mood', project='g2net')
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# trainer
trainer = Trainer(**dict(cfg.trainer), callbacks=[lr_monitor], logger=logger)

# Fit and test
trainer.fit(litmodel, dm)
trainer.test(litmodel)
