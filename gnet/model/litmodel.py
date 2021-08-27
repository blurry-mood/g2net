import pytorch_lightning as pl
import torch

from .model import model
from ..utils import get_logger
from ..preprocessing.preprocesser import Preprocessor

from torch import nn
from deepblocks.loss import FocalLoss
from torchmetrics import AUROC
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

_logger = get_logger()

_LOSS = {'celoss': nn.CrossEntropyLoss,
         'focalloss': FocalLoss, 'bceloss': nn.BCEWithLogitsLoss}
_OPT = {'adamw': AdamW, 'adam': Adam, 'sgd': SGD}
_SCHEDULER = {'linear': get_linear_schedule_with_warmup,
              'step': StepLR, 'cosine': get_cosine_schedule_with_warmup}


class LitModel(pl.LightningModule):

    def __init__(self, config, preprocess_config_name):
        super().__init__()

        self.save_hyperparameters()

        self.config = config
        self.preprocess = Preprocessor(preprocess_config_name)
        self.model = model(config.model_name,
                           config.pretrained, config.num_classes)

        # choose loss
        self.loss = _LOSS[config.loss.name]
        if config.loss.args:
            self.loss = self.loss(**dict(config.loss.args))
        else:
            self.loss = self.loss()

        # metric
        self.train_auc = AUROC(config.num_classes, compute_on_step=True)
        self.val_auc = AUROC(config.num_classes, compute_on_step=False)

        # log
        _logger.info('The model is created')

    def configure_optimizers(self):
        opt = _OPT[self.config.optimizer.name](
            self.parameters(), **dict(self.config.optimizer.args))
        scheduler = _SCHEDULER[self.config.scheduler.name](
            opt, **dict(self.config.scheduler.args))
        return [opt], [scheduler]

    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)
        if isinstance(self.loss, nn.BCELoss):
            probs = torch.sigmoid(y_hat)
        else:
            probs = torch.softmax(y_hat, dim=1)

        if len(y.unique()) != 1:
            auc = self.train_auc(probs, y)
            self.log('train_auc', auc, prog_bar=True)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        if isinstance(self.loss, nn.BCELoss):
            probs = torch.sigmoid(y_hat)
        else:
            probs = torch.softmax(y_hat, dim=1)

        self.val_auc(probs, y)
            
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def validation_epoch_end(self, outs):
        self.log('val_auc', self.val_auc.compute(), prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        if isinstance(self.loss, nn.BCELoss):
            probs = torch.sigmoid(y_hat)
        else:
            probs = torch.softmax(y_hat, dim=1)

        self.val_auc(probs, y)

        self.log('test_loss', loss, prog_bar=True)

        return loss

    def test_epoch_end(self, outputs) -> None:
        self.log('test_auc', self.val_auc.compute(), prog_bar=True)
