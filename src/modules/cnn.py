import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
import numpy as np


class SingleNetworkCNNModule(pl.LightningModule):

    def __init__(self, lr: float, embeddings: np.array, hidden_dim: int, output_labels: int, p: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        embedding_dim = embeddings.shape[1]
        out_channel = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embeddings))
        self.conv2 = nn.Conv1d(embedding_dim, out_channel, kernel_size=3)
        self.conv3 = nn.Conv1d(42, out_channel, kernel_size=3)
        # self.conv4  = nn.Conv1d(embedding_dim,out_channel, kernel_size = 4)
        # self.conv5  = nn.Conv1d(embedding_dim,out_channel, kernel_size = 5)
        # self.conv6  = nn.Conv1d(embedding_dim,out_channel, kernel_size = 6)
        # The linear layer that maps from hidden state space to tag space
        self.output = nn.Linear(16 * 4, output_labels)
        self.dropout = nn.Dropout(p)

        # hyperparams
        self.lr = lr
        self.p = p

    def forward(self, sentence1):
        output = self.embedding(sentence1)
        output = output.permute(0, 2, 1)
        output = F.max_pool1d(self.conv2(output), kernel_size=3)
        output = output.permute(0, 2, 1)
        output = F.max_pool1d(self.conv3(output), kernel_size=3)

        output = self.dropout(self.output(F.dropout(output.flatten(start_dim=1), p=self.p)))
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


class RankedNetworkCNNModule(pl.LightningModule):

    def __init__(self, lr: float, embeddings: np.array, hidden_dim: int, output_labels: int, p: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        embedding_dim = embeddings.shape[1]
        out_channel = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embeddings))
        self.conv1 = nn.Conv1d(embedding_dim, out_channel, kernel_size=3)
        self.conv3 = nn.Conv1d(embedding_dim, out_channel, kernel_size=3)
        self.conv2 = nn.Conv1d(42, out_channel, kernel_size=3)
        self.conv4 = nn.Conv1d(42, out_channel, kernel_size=3)
        # self.conv4  = nn.Conv1d(embedding_dim,out_channel, kernel_size = 4)
        # self.conv5  = nn.Conv1d(embedding_dim,out_channel, kernel_size = 5)
        # self.conv6  = nn.Conv1d(embedding_dim,out_channel, kernel_size = 6)
        # The linear layer that maps from hidden state space to tag space
        self.output = nn.Linear(16 * 4  * 2, output_labels)
        self.dropout = nn.Dropout(p)

        # hyperparams
        self.lr = lr
        self.p = p

    def forward(self, sentence1, sentence2):
        output1 = self.embedding(sentence1)
        output1 = output1.permute(0, 2, 1)
        output1 = F.max_pool1d(self.conv1(output1), kernel_size=3)
        output1 = output1.permute(0, 2, 1)
        output1 = F.max_pool1d(self.conv2(output1), kernel_size=3)

        output2 = self.embedding(sentence2)
        output2 = output2.permute(0, 2, 1)
        output2 = F.max_pool1d(self.conv3(output2), kernel_size=3)
        output2 = output2.permute(0, 2, 1)
        output2 = F.max_pool1d(self.conv4(output2), kernel_size=3)

        output = self.dropout(
            self.output(F.dropout(torch.cat((output1.flatten(start_dim=1), output2.flatten(start_dim=1)), axis=1),
                                  p=self.p)))
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
