import argparse
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import wandb
import os


def train(cfg_name, data_path):
    cfg = os.path.join(os.path.split(__file__)[0], 'config', cfg_name+'.yaml')
    cfg = OmegaConf.load(cfg)

    # model & datamodule
    litmodel = LitModel(cfg)
    dm = DataModule(data_path, cfg.datamodule, cfg.split)

    # wandb logger & lr monitor
    logger = WandbLogger(entity='blurry-mood', project='g2net')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # trainer
    trainer = Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        **dict(cfg.trainer),
        callbacks=[lr_monitor], logger=logger)

    # Fit and test
    trainer.fit(litmodel, dm)
    trainer.test(litmodel)
    wandb.finish(0)


if __name__ == '__main__':
    from gnet.dataset.datamodules import DataModule
    from gnet.model.litmodel import LitModel

    # arguments
    parser = argparse.ArgumentParser(description='Train Lightning Module.')
    parser.add_argument('--yaml', type=str, required=True,
                        help='YAML config file containing the lit model description.')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training dataset. The training folder should contain train_labels.csv with keys [id, target]. Wildcards are supported.')
    args = parser.parse_args()

    cfg = args.yaml
    data_path = args.data

    train(cfg, data_path)

else:
    from ..dataset.datamodules import DataModule
    from .litmodel import LitModel
