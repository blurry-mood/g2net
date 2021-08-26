import pytorch_lightning as pl
import torch

from .model import model
from ..utils import get_logger

from torch import nn
from deepblocks.loss import FocalLoss
from torchmetrics import AUROC
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

_logger = get_logger()

_LOSS = {'celoss': nn.CrossEntropyLoss, 'focalloss': FocalLoss, 'bceloss':nn.BCEWithLogitsLoss}
_OPT = {'adamw': AdamW, 'adam': Adam, 'sgd': SGD}
_SCHEDULER = {'linear': get_linear_schedule_with_warmup,
              'step': StepLR, 'cosine': get_cosine_schedule_with_warmup}


class LitModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model = model(config.model_name,
                           config.pretrained, config.num_classes)
        self.loss = _LOSS[config.loss.name]
        if config.loss.args:
            self.loss = self.loss(**dict(config.loss.args))
        else:
            self.loss = self.loss()
            
        self.auc = AUROC(config.num_classes, compute_on_step=True)
        _logger.info('The model is created')

    def configure_optimizers(self):
        opt = _OPT[self.config.optimizer.name](
            self.parameters(), **dict(self.config.optimizer.args))
        scheduler = _SCHEDULER[self.config.scheduler.name](
            opt, **dict(self.config.scheduler.args))
        return [opt], [scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.loss(y_hat, y)
        # if isinstance(self.loss, nn.BCELoss):
        #     probs = torch.sigmoid(y_hat)
        # else:
        probs = torch.softmax(y_hat, dim=1)
        auc = self.auc(probs, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_auc', auc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        # if isinstance(self.loss, nn.BCELoss):
        #     probs = torch.sigmoid(y_hat)
        # else:
        probs = torch.softmax(y_hat, dim=1)
        auc = self.auc(probs, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_auc', auc, prog_bar=True)

        return loss 

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        # if isinstance(self.loss, nn.BCELoss):
        #     probs = torch.sigmoid(y_hat)
        # else:
        probs = torch.softmax(y_hat, dim=1)
        auc = self.auc(probs, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_auc', auc, prog_bar=True)

        return loss 