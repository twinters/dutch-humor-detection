import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
import numpy as np


class SingleNetworkLSTMModule(pl.LightningModule):

    def __init__(self, lr: float, embeddings: np.array, hidden_dim: int, output_labels: int, p: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embeddings))
        self.h1 = nn.LSTM(embeddings.shape[1], hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.output = nn.Linear(hidden_dim, output_labels)
        self.dropout = nn.Dropout(p)

        # hyperparams
        self.lr = lr
        self.p = p

    def forward(self, sentence1):
        outputs, (ht, ct) = self.h1(self.embedding(sentence1))
        output = self.dropout(self.output(F.dropout(ht[-1], p=self.p)))
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        y_hat = self(batch['input_ids'])
        loss = nn.CrossEntropyLoss()(y_hat, batch['label'].flatten())
        result = pl.TrainResult(loss)
        return result

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch['input_ids'])
        loss = nn.CrossEntropyLoss()(y_hat, batch['label'].flatten())
        acc = FM.accuracy(y_hat.detach().argmax(axis=1), batch['label'].flatten(), num_classes=2)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_acc': acc, 'val_loss': loss}, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        y_hat = self(batch['input_ids'])
        loss = nn.CrossEntropyLoss()(y_hat, batch['label'].flatten())
        acc = FM.accuracy(y_hat.detach().argmax(axis=1), batch['label'].flatten(), num_classes=2)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_acc': acc, 'val_loss': loss})
        return result


class RankedNetworkLSTMModule(pl.LightningModule):

    def __init__(self, lr: float, embeddings: np.array, hidden_dim: int, output_labels: int, p: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embeddings))
        self.h1 = nn.LSTM(embeddings.shape[1], hidden_dim, batch_first=True)
        self.h2 = nn.LSTM(embeddings.shape[1], hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.output = nn.Linear(hidden_dim * 2, output_labels)
        self.dropout = nn.Dropout(p)

        # hyperparams
        self.lr = lr
        self.p = p

    def forward(self, sentence1, sentence2):
        outputs1, (ht1, ct1) = self.h1(self.embedding(sentence1))
        outputs2, (ht2, ct2) = self.h2(self.embedding(sentence2))
        ht1 = F.dropout(ht1, p=self.p)
        ht2 = F.dropout(ht2, p=self.p)
        output = self.dropout(self.output(torch.cat((ht1[-1], ht2[-1]), axis=1)))
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        y_hat = self(batch['a'], batch['b'])
        loss = nn.CrossEntropyLoss()(y_hat, batch['label'].flatten())
        result = pl.TrainResult(loss)
        return result

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch['a'], batch['b'])
        loss = nn.CrossEntropyLoss()(y_hat, batch['label'].flatten())
        acc = FM.accuracy(y_hat.detach().argmax(axis=1), batch['label'].flatten(), num_classes=2)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_acc': acc, 'val_loss': loss}, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        y_hat = self(batch['a'], batch['b'])
        loss = nn.CrossEntropyLoss()(y_hat, batch['label'].flatten())
        acc = FM.accuracy(y_hat.detach().argmax(axis=1), batch['label'].flatten(), num_classes=2)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_acc': acc, 'val_loss': loss})
        return result

