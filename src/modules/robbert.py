from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

from src.modules.GaussianNaiveBayes import GaussianNaiveBayes
from transformers import AutoModelForSequenceClassification, AutoConfig, AdamW


class RobBERTModule(pl.LightningModule):

    def __init__(self, hparams: Dict, model):
        super().__init__()

        # hyperparams
        self.hparams = hparams

        self.model = model

    def forward(self, sentence1):

        return self.model(**sentence1)

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        return optimizer

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = loss = outputs[0]
        result = pl.TrainResult(loss)
        return result

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = loss = outputs[0]
        y_hat = outputs[1]
        acc = FM.accuracy(y_hat.detach().argmax(axis=1), batch['labels'].flatten(), num_classes=2)
        result = pl.EvalResult(checkpoint_on=loss)
        #result.log_dict({'val_acc': acc, 'val_loss': loss}, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs[0]
        y_hat = outputs[1]
        acc = FM.accuracy(y_hat.detach().argmax(axis=1), batch['labels'].flatten(), num_classes=2)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_acc': acc, 'val_loss': loss})
        return result
