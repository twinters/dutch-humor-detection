import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM


class SingleNetworkModule(pl.LightningModule):

    def __init__(self, lr: float, embedding_dim: int, hidden_dim: int, output_labels: int, p: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.h1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(p)

        # The linear layer that maps from hidden state space to tag space
        self.output = nn.Linear(hidden_dim, output_labels)

        # hyperparams
        self.lr = lr

    def forward(self, sentence1):
        h1_out = F.relu(self.dropout(self.h1(sentence1)))
        output = self.output(h1_out)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        y_hat = self(batch['sentence1'])
        loss = nn.CrossEntropyLoss()(y_hat, batch['label'].flatten())
        result = pl.TrainResult(loss)
        return result

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch['sentence1'])
        loss = nn.CrossEntropyLoss()(y_hat, batch['label'].flatten())
        acc = FM.accuracy(y_hat.detach().argmax(axis=1), batch['label'].flatten(), num_classes=2)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_acc': acc, 'val_loss': loss}, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        y_hat = self(batch['sentence1'])
        loss = nn.CrossEntropyLoss()(y_hat, batch['label'].flatten())
        acc = FM.accuracy(y_hat.detach().argmax(axis=1), batch['label'].flatten(), num_classes=2)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_acc': acc, 'val_loss': loss})
        return result

class DualNetworkModule(pl.LightningModule):

    def __init__(self, lr: float, embedding_dim: int, hidden_dim: int, output_labels: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # The RNNs for the left and right parts
        self.lstm1 = nn.Linear(embedding_dim, hidden_dim)
        self.lstm2 = nn.Linear(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.output = nn.Linear(hidden_dim * 2, output_labels)

        # hyperparams
        self.lr = lr

    def forward(self, sentence1, sentence2):
        lstm1_out = F.relu(self.lstm1(sentence1))
        lstm2_out = F.relu(self.lstm2(sentence2))
        output = self.output(torch.cat((lstm1_out, lstm2_out), dim=1))
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        y_hat = self(batch['sentence1'], batch['sentence2'])
        loss = nn.CrossEntropyLoss()(y_hat, batch['label'].flatten())
        result = pl.TrainResult(loss)
        return result

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch['sentence1'], batch['sentence2'])
        loss = nn.CrossEntropyLoss()(y_hat, batch['label'].flatten())
        acc = FM.accuracy(y_hat.detach().argmax(axis=1), batch['label'].flatten(), num_classes=2)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_acc': acc, 'val_loss': loss}, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        y_hat = self(batch['sentence1'], batch['sentence2'])
        loss = nn.CrossEntropyLoss()(y_hat, batch['label'].flatten())
        acc = FM.accuracy(y_hat.detach().argmax(axis=1), batch['label'].flatten(), num_classes=2)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_acc': acc, 'val_loss': loss})
        return result
