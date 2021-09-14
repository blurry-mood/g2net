import pytorch_lightning as pl
import torch

from .model import model
from ..utils import get_logger
from ..preprocessing.preprocesser import Preprocessor

from torch import nn
from deepblocks.loss import FocalLoss, AUCLoss
from torchmetrics import AUROC, Accuracy, F1
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

_logger = get_logger()

_LOSS = {'celoss': nn.CrossEntropyLoss,
         'focalloss': FocalLoss, 'bceloss': nn.BCEWithLogitsLoss, 'aucloss': AUCLoss}
_OPT = {'adamw': AdamW, 'adam': Adam, 'sgd': SGD}
_SCHEDULER = {'linear': get_linear_schedule_with_warmup,
              'step': StepLR, 'cosine': get_cosine_schedule_with_warmup}


class BinaryLitModel(pl.LightningModule):

    def __init__(self, config, preprocess_config_name):
        super().__init__()

        self.save_hyperparameters()

        self.config = config
        self.max_epochs = config.trainer.max_epochs*2
        self.preprocess = Preprocessor(preprocess_config_name)
        self.model = model(config.model_name,
                           config.pretrained, 1)
        self.show_shape = True

        # choose loss
        self.loss = _LOSS[config.loss.name]
        if config.loss.args:
            self.loss = self.loss(**dict(config.loss.args))
        else:
            self.loss = self.loss()

        # metric
        self.train_auroc = AUROC(pos_label=1, compute_on_step=True)
        self.val_auroc = AUROC(pos_label=1, compute_on_step=False)
        self.val_acc = Accuracy(num_classes=1, compute_on_step=True, )
        self.val_f1 = F1(compute_on_step=True, num_classes=1, )

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

        loss = self.loss(y_hat, y.unsqueeze(1).float())

        probs = torch.sigmoid(y_hat)

        if len(y.unique()) != 1:
            auc = self.train_auroc(probs, y.unsqueeze(1))
            self.log('train_auroc', auc, prog_bar=True)

        self.log('train_loss', loss, prog_bar=True)
        lmd = (self.trainer.current_epoch/self.max_epochs)**2
        loss = loss - lmd*y_hat.abs().mean()
        _logger.info(f'Current lambda is {lmd:.4f}')

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.show_shape:
            self.show_shape = False
            _logger.info(
                f'Raw input shape: {x.shape}, mean: {x.mean()}, std: {x.std()}')
            xx = self.preprocess(x)
            _logger.info(
                f'Preprocessed input shape: {xx.shape}, mean: {xx.mean()}, std: {xx.std()}')
        loss = self.loss(y_hat, y.unsqueeze(1).float())

        probs = torch.sigmoid(y_hat)

        self.val_auroc(probs, y.unsqueeze(1))

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc(probs, y.unsqueeze(1)), prog_bar=True)
        self.log('val_f1', self.val_f1(probs, y.unsqueeze(1)), prog_bar=True)
        return loss

    def validation_epoch_end(self, outs):
        self.log('val_auroc', self.val_auroc.compute(), prog_bar=True)
        self.val_auroc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.unsqueeze(1).float())

        probs = torch.sigmoid(y_hat)

        self.val_auroc(probs, y.unsqueeze(1))

        self.log('test_loss', loss, prog_bar=True, on_step=True)

        self.log('test_acc', self.val_acc(probs, y.unsqueeze(1)), prog_bar=True)
        self.log('test_f1', self.val_f1(probs, y.unsqueeze(1)), prog_bar=True)
        return loss

    def test_epoch_end(self, outputs) -> None:
        self.log('test_auroc', self.val_auroc.compute(), prog_bar=True)
        self.val_auroc.reset()


class MultiLitModel(pl.LightningModule):

    def __init__(self, config, preprocess_config_name):
        super().__init__()

        self.save_hyperparameters()

        self.config = config
        self.max_epochs = config.trainer.max_epochs*2
        self.preprocess = Preprocessor(preprocess_config_name)
        self.model = model(config.model_name,
                           config.pretrained, 2)
        self.show_shape = True

        # choose loss
        self.loss = _LOSS[config.loss.name]
        if config.loss.args:
            self.loss = self.loss(**dict(config.loss.args))
        else:
            self.loss = self.loss()

        # metric
        self.train_auroc = AUROC(pos_label=1, compute_on_step=True)
        self.val_auroc = AUROC(pos_label=1, compute_on_step=False)
        self.val_acc = Accuracy(num_classes=1, compute_on_step=True, )
        self.val_f1 = F1(compute_on_step=True, num_classes=1,)

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

        probs = torch.softmax(y_hat, dim=1)

        if len(y.unique()) != 1:
            auc = self.train_auroc(probs[:,1], y)
            self.log('train_auroc', auc, prog_bar=True)

        self.log('train_loss', loss, prog_bar=True)

        lmd = (self.trainer.current_epoch/self.max_epochs)**4
        loss = loss - lmd*y_hat.abs().mean()
        _logger.info(f'Current lambda is {lmd:.4f}')

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.show_shape:
            self.show_shape = False
            _logger.info(
                f'Raw input shape: {x.shape}, mean: {x.mean()}, std: {x.std()}')
            xx = self.preprocess(x)
            _logger.info(
                f'Preprocessed input shape: {xx.shape}, mean: {xx.mean()}, std: {xx.std()}')

        loss = self.loss(y_hat, y)

        probs = torch.softmax(y_hat, dim=1)

        self.val_auroc(probs[:,1], y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc(probs[:,1], y), prog_bar=True)
        self.log('val_f1', self.val_f1(probs[:,1], y), prog_bar=True)

        return loss

    def trainig_epoch_end(self, args):
        self.train_auroc.reset()

    def validation_epoch_end(self, outs):
        self.log('val_auroc', self.val_auroc.compute(), prog_bar=True)
        self.val_auroc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        probs = torch.softmax(y_hat, dim=1)

        self.val_auroc(probs[:,1], y)

        self.log('test_loss', loss, prog_bar=True, on_step=True)
        self.log('test_acc', self.val_acc(probs[:,1], y), prog_bar=True)
        self.log('test_f1', self.val_f1(probs[:,1], y), prog_bar=True)

        return loss

    def test_epoch_end(self, outputs) -> None:
        self.log('test_auroc', self.val_auroc.compute(), prog_bar=True)
        self.val_auroc.reset()


class PredictLitModel(pl.LightningModule):

    def __init__(self, config, preprocess_config_name):
        super().__init__()

        self.config = config
        self.preprocess = Preprocessor(preprocess_config_name)
        self.model = model(config.model_name,
                           config.pretrained, config.num_classes)
        self.softmax = config.num_classes == 2

        # log
        _logger.info('The model is created')

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x)
        if self.softmax:
            x = torch.softmax(x, dim=1)[:, 1:]
        else:
            x = torch.sigmoid(x)
        return x[:,0]