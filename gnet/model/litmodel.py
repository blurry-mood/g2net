import pytorch_lightning as pl
import torch

from .model import model, Paper
from ..utils import get_logger
from ..preprocessing.preprocesser import Preprocessor

from torch import nn
from torch.nn.functional import softmax,  log_softmax

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

class LitModel(pl.LightningModule):

    def __init__(self, config, preprocess_config_name):
        super().__init__()

        self.save_hyperparameters()

        self.preprocess = Preprocessor(preprocess_config_name)
        self.model = model(config.model_name,
                           config.pretrained, config.num_classes)
        
        self.multi_cls = config.num_classes>1
        self.show_shape = True

        # choose loss
        self.loss = _LOSS[config.loss.name]
        if config.loss.args:
            self.loss = self.loss(**dict(config.loss.args))
        else:
            self.loss = self.loss()

        # metric
        self.train_auroc = AUROC(compute_on_step=False)
        self.val_auroc = AUROC(compute_on_step=False)
        self.val_acc = Accuracy(compute_on_step=True, )
        self.val_f1 = F1(compute_on_step=True, )

        # log
        _logger.info('The model is created')

    def configure_optimizers(self):
        opt = _OPT[self.hparams.config.optimizer.name](
            self.parameters(), **dict(self.hparams.config.optimizer.args))
        scheduler = _SCHEDULER[self.hparams.config.scheduler.name](
            opt, **dict(self.hparams.config.scheduler.args))
        return [opt], [scheduler]

    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if not self.multi_cls:
            y = y.unsqueeze(1)

        probs = torch.softmax(y_hat, dim=1)[:, 1] if self.multi_cls else torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y if self.multi_cls else y.float() )
        self.train_auroc(probs, y)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.show_shape:
            self.show_shape = False
            _logger.info(
                f'Raw input shape: {x.shape}, mean: {x.mean()}, std: {x.std()}')
            xx = self.preprocess(x)
            _logger.info(
                f'Preprocessed input shape: {xx.shape}, mean: {xx.mean()}, std: {xx.std()}')
        
        y_hat = self(x)


        if not self.multi_cls:
            y = y.unsqueeze(1)

        probs = torch.softmax(y_hat, dim=1)[:, 1] if self.multi_cls else torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y if self.multi_cls else y.float() )
        self.val_auroc(probs, y)

        # logs
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc(probs, y), prog_bar=True)
        self.log('val_f1', self.val_f1(probs, y), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if not self.multi_cls:
            y = y.unsqueeze(1)

        probs = torch.softmax(y_hat, dim=1)[:, 1] if self.multi_cls else torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y if self.multi_cls else y.float() )
        self.val_auroc(probs, y)

        # logs
        self.log('test_loss', loss, prog_bar=True, on_step=True)
        self.log('test_acc', self.val_acc(probs, y), prog_bar=True)
        self.log('test_f1', self.val_f1(probs, y), prog_bar=True)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log('train_auroc', self.train_auroc.compute(), prog_bar=True)
        self.train_auroc.reset()

    def validation_epoch_end(self, outputs):
        self.log('val_auroc', self.val_auroc.compute(), prog_bar=True)
        self.val_auroc.reset()

    def test_epoch_end(self, outputs) -> None:
        self.log('test_auroc', self.val_auroc.compute(), prog_bar=True)
        self.val_auroc.reset()


class PaperLitModel(pl.LightningModule):

    def __init__(self, config, preprocess_config_name):
        super().__init__()

        self.save_hyperparameters()

        self.preprocess = Preprocessor(preprocess_config_name)
        self.model = Paper(config.num_classes)
        
        self.multi_cls = config.num_classes>1
        self.show_shape = True

        # choose loss
        self.loss = _LOSS[config.loss.name]
        if config.loss.args:
            self.loss = self.loss(**dict(config.loss.args))
        else:
            self.loss = self.loss()

        # metric
        self.train_auroc = AUROC(compute_on_step=False)
        self.val_auroc = AUROC(compute_on_step=False)
        self.val_acc = Accuracy(compute_on_step=True, )
        self.val_f1 = F1(compute_on_step=True, )

        # log
        _logger.info('The model is created')

    def configure_optimizers(self):
        opt = _OPT[self.hparams.config.optimizer.name](
            self.parameters(), **dict(self.hparams.config.optimizer.args))
        scheduler = _SCHEDULER[self.hparams.config.scheduler.name](
            opt, **dict(self.hparams.config.scheduler.args))
        return [opt], [scheduler]

    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if not self.multi_cls:
            y = y.unsqueeze(1)

        probs = torch.softmax(y_hat, dim=1)[:, 1] if self.multi_cls else torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y if self.multi_cls else y.float() )
        self.train_auroc(probs, y)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.show_shape:
            self.show_shape = False
            _logger.info(
                f'Raw input shape: {x.shape}, mean: {x.mean()}, std: {x.std()}')
            xx = self.preprocess(x)
            _logger.info(
                f'Preprocessed input shape: {xx.shape}, mean: {xx.mean()}, std: {xx.std()}')
        
        y_hat = self(x)


        if not self.multi_cls:
            y = y.unsqueeze(1)

        probs = torch.softmax(y_hat, dim=1)[:, 1] if self.multi_cls else torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y if self.multi_cls else y.float() )
        self.val_auroc(probs, y)

        # logs
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc(probs, y), prog_bar=True)
        self.log('val_f1', self.val_f1(probs, y), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if not self.multi_cls:
            y = y.unsqueeze(1)

        probs = torch.softmax(y_hat, dim=1)[:, 1] if self.multi_cls else torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y if self.multi_cls else y.float() )
        self.val_auroc(probs, y)

        # logs
        self.log('test_loss', loss, prog_bar=True, on_step=True)
        self.log('test_acc', self.val_acc(probs, y), prog_bar=True)
        self.log('test_f1', self.val_f1(probs, y), prog_bar=True)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log('train_auroc', self.train_auroc.compute(), prog_bar=True)
        self.train_auroc.reset()

    def validation_epoch_end(self, outputs):
        self.log('val_auroc', self.val_auroc.compute(), prog_bar=True)
        self.val_auroc.reset()

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

def kl_div(x, y):
    px = softmax(x, dim=1)
    lpx, lpy = log_softmax(x, dim=1), log_softmax(y, dim=1)
    return (px*(lpx-lpy)).mean()

class DMLLitModel(pl.LightningModule):

    def __init__(self, config, preprocess_config_name):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False

        self.preprocess = Preprocessor(preprocess_config_name)
        self.student1 = model(config.model_name,
                           config.pretrained1, config.num_intermediate)
        self.student2 = model(config.model_name,
                           config.pretrained2, config.num_intermediate)
        self.linear1 = nn.Linear(config.num_intermediate, config.num_classes)
        self.linear2 = nn.Linear(config.num_intermediate, config.num_classes)

        self.multi_cls = config.num_classes>1
        self.show_shape = True

        # choose loss
        self.loss = _LOSS[config.loss.name]
        if config.loss.args:
            self.loss = self.loss(**dict(config.loss.args))
        else:
            self.loss = self.loss()

        # metric
        self.train_auroc1 = AUROC(compute_on_step=False)
        self.val_auroc1 = AUROC(compute_on_step=False)
        self.train_auroc2 = AUROC(compute_on_step=False)
        self.val_auroc2 = AUROC(compute_on_step=False)

        # log
        _logger.info('The model is created')

    def configure_optimizers(self):
        opt = _OPT[self.hparams.config.optimizer.name](
            # [*self.student1.parameters(), *self.student2.parameters(), *self.linear1.parameters(), *self.linear2.parameters()],
            self.parameters(),
             **dict(self.hparams.config.optimizer.args))
        scheduler = _SCHEDULER[self.hparams.config.scheduler.name](
            opt, **dict(self.hparams.config.scheduler.args))
        return [opt], [scheduler]

    def forward(self, x, optimize_first:bool=True):
        x = self.preprocess(x)
        if optimize_first:
            with torch.no_grad():
                x2 = self.student2(x)
                y2 = self.linear2(x2)
            x1 = self.student1(x)
            y1 = self.linear1(x1)
        else:
            with torch.no_grad():
                x1 = self.student1(x)
                y1 = self.linear1(x1)
            x2 = self.student2(x)
            y2 = self.linear2(x2)
        return (x1, y1), (x2, y2)

    def training_step(self, batch, batch_idx):
        x, y = batch

        opt = self.optimizers()

        # Optimize student 1
        (x1, y1), (x2, y2) = self(x)
        if not self.multi_cls:
            y1 = y1.unsqueeze(1)

        loss = self.loss(y1, y if self.multi_cls else y.float()) + 0.5*kl_div(x2, x1)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.log('train_loss1', loss, prog_bar=True,)

        probs = torch.softmax(y1, dim=1)[:, 1] if self.multi_cls else torch.sigmoid(y1)
        self.train_auroc1(probs, y)
        
        # Optimize student 2
        (x1, y1), (x2, y2) = self(x, False)
        if not self.multi_cls:
            y2 = y2.unsqueeze(1)

        loss = self.loss(y2, y if self.multi_cls else y.float()) + 0.5*kl_div(x1, x2)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.log('train_loss2', loss, prog_bar=True,)

        probs = torch.softmax(y2, dim=1)[:, 1] if self.multi_cls else torch.sigmoid(y2)
        self.train_auroc2(probs, y)
        
        if self.trainer.is_last_batch :
            self.lr_schedulers().step()


    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.show_shape:
            self.show_shape = False
            _logger.info(
                f'Raw input shape: {x.shape}, mean: {x.mean()}, std: {x.std()}')
            xx = self.preprocess(x)
            _logger.info(
                f'Preprocessed input shape: {xx.shape}, mean: {xx.mean()}, std: {xx.std()}')

        (x1, y1), (x2, y2) = self(x)

        if not self.multi_cls:
            y1 = y1.unsqueeze(1)
            y2 = y2.unsqueeze(1)

        loss1 = self.loss(y1, y if self.multi_cls else y.float())
        loss2 = self.loss(y2, y if self.multi_cls else y.float())
        
        self.log('val_loss1', loss1, prog_bar=True,)
        self.log('val_loss2', loss2, prog_bar=True,)

        probs1 = torch.softmax(y1, dim=1)[:, 1] if self.multi_cls else torch.sigmoid(y1)
        probs2 = torch.softmax(y2, dim=1)[:, 1] if self.multi_cls else torch.sigmoid(y2)
        
        self.val_auroc1(probs1, y)
        self.val_auroc2(probs2, y)
        
        return loss2

    def test_step(self, batch, batch_idx):
        x, y = batch
        (x1, y1), (x2, y2) = self(x)

        if not self.multi_cls:
            y1 = y1.unsqueeze(1)
            y2 = y2.unsqueeze(1)

        loss1 = self.loss(y1, y if self.multi_cls else y.float())
        loss2 = self.loss(y2, y if self.multi_cls else y.float())
        
        self.log('test_loss1', loss1, prog_bar=True,)
        self.log('test_loss2', loss2, prog_bar=True,)

        probs1 = torch.softmax(y1, dim=1)[:, 1] if self.multi_cls else torch.sigmoid(y1)
        probs2 = torch.softmax(y2, dim=1)[:, 1] if self.multi_cls else torch.sigmoid(y2)
        
        self.val_auroc1(probs1, y)
        self.val_auroc2(probs2, y)
        
        return loss2

    def training_epoch_end(self, outputs) -> None:
        self.log('train_auroc1', self.train_auroc1.compute(), prog_bar=True)
        self.train_auroc1.reset()
        self.log('train_auroc2', self.train_auroc2.compute(), prog_bar=True)
        self.train_auroc2.reset()

    def validation_epoch_end(self, outputs):
        self.log('val_auroc1', self.val_auroc1.compute(), prog_bar=True)
        self.val_auroc1.reset()
        self.log('val_auroc2', self.val_auroc2.compute(), prog_bar=True)
        self.val_auroc2.reset()

    def test_epoch_end(self, outputs) -> None:
        self.log('test_auroc1', self.val_auroc1.compute(), prog_bar=True)
        self.val_auroc1.reset()
        self.log('test_auroc2', self.val_auroc2.compute(), prog_bar=True)
        self.val_auroc2.reset()