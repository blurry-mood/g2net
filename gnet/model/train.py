from glob import glob
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import wandb
import os

from .litmodel import LitModel, DMLLitModel, PaperLitModel
from ..loader.datamodule import DataModule
from ..utils import get_logger


os.environ["WANDB_START_METHOD"] = "thread"

torch.backends.cudnn.benchmark = True

_HERE = os.path.split(__file__)[0]
_logger = get_logger()

_LITMODELS = {'standard': LitModel, 'dml': DMLLitModel, 'paper':PaperLitModel}

def train(model_cfg_name, pre_cfg_name, dm_cfg_name, data_path):
    cfg = glob(os.path.join(_HERE, 'config', '**', model_cfg_name+'.yaml'), recursive=True)
    cfg = OmegaConf.load(cfg[0])
    
    # model & datamodule
    litmodel = _LITMODELS[cfg.type](cfg, pre_cfg_name)
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
    
    # Fit
    trainer.fit(litmodel, dm)

    # Test
    trainer.test(litmodel)
    
    # clean output folder, then save only the ckpt file
    dirs = glob('*')
    if not 'gnet' in dirs:      # to avoid executing this locally
        _logger.info('Cleaning output folder...')
        os.system('rm * -rf')
    trainer.save_checkpoint("litmodel.ckpt")

    # log artifacts
    # run = trainer.logger.experiment
    # artifact = wandb.Artifact('my-model', type='model')
    # artifact.add_file('litmodel.ckpt')
    # run.log_artifact(artifact)
    
    # artifact = wandb.Artifact('my-code', type='dataset')
    # artifact.add_dir(os.path.join(_HERE, '..'))
    # run.log_artifact(artifact)


def successive_train(*model_cfg_names, pre_cfg_name, dm_cfg_name, data_path):
    dm = DataModule(data_path, dm_cfg_name)

    def configuration(path, second:bool=False):
        
        cfg = glob(os.path.join(_HERE, 'config', '**', path+'.yaml'), recursive=True)
        cfg = OmegaConf.load(cfg[0])
        
        # model & datamodule
        if second:
            litmodel = _LITMODELS[cfg.type].load_from_checkpoint('litmodel.ckpt', config=cfg, preprocess_config_name=pre_cfg_name)
        else:
            litmodel = _LITMODELS[cfg.type](cfg, pre_cfg_name)

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
        
        return trainer, litmodel
        
    """ First round"""
    trainer, litmodel = configuration(model_cfg_names[0])
    trainer.fit(litmodel, dm)
    trainer.test(litmodel)
    
    # clean output folder, then save only the ckpt file
    dirs = glob('*')
    if not 'gnet' in dirs:      # to avoid executing this locally
        _logger.info('Cleaning output folder...')
        os.system('rm * -rf')
    trainer.save_checkpoint("litmodel.ckpt")

    """ Second round"""
    for i in range(1, len(model_cfg_names)):
        trainer, litmodel =  configuration(model_cfg_names[i], True)
        trainer.fit(litmodel, dm)
        trainer.test(litmodel)

        # clean output folder, then save only the ckpt file
        dirs = glob('*')
        if not 'gnet' in dirs:      # to avoid executing this locally
            _logger.info('Cleaning output folder...')
            os.system('rm * -rf')
        trainer.save_checkpoint("litmodel.ckpt")