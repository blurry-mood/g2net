import argparse
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import wandb
import os

from .litmodel import BinaryLitModel, MultiLitModel
from ..loader.datamodule import DataModule

torch.backends.cudnn.benchmark = True

_HERE = os.path.split(__file__)[0]

def train(model_cfg_name, pre_cfg_name, dm_cfg_name, data_path):
    cfg = os.path.join(_HERE, 'config', model_cfg_name+'.yaml')
    cfg = OmegaConf.load(cfg)

    # model & datamodule
    litmodel = BinaryLitModel(cfg, pre_cfg_name) if cfg.num_classes==1 else MultiLitModel(cfg, pre_cfg_name)
    dm = DataModule(data_path, dm_cfg_name)

    # wandb logger & lr monitor
    logger = WandbLogger(entity='blurry-mood', project='g2net')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # trainer
    trainer = Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        **dict(cfg.trainer),
        callbacks=[lr_monitor], 
        logger=logger
        )

    # Fit and test
    trainer.fit(litmodel, dm)
    trainer.test(litmodel)
    
    # push to cloud
    wandb.finish(0)