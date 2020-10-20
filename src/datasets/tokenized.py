import json
import random

import numpy as np
import pandas as pd
import typing

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ClassificationDataset(Dataset):

    def __init__(self, original: str, replaced: str, tokenizer: PreTrainedTokenizer):
        with open(original) as fp:
            self.df = pd.DataFrame(json.load(fp), columns=["text"])
            self.df['label'] = 0

        with open(replaced) as fp:
            self.df2 = pd.DataFrame(json.load(fp), columns=['text'])
            self.df2['label'] = 1

            self.combined = self.df.append(self.df2, ignore_index=True)
            self.combined = self.combined.sample(frac=1.0)  # Shuffle

            self.tokenizer = tokenizer

    def get_embedding_size(self):
        return 1

    def __len__(self):
        return len(self.combined)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.tokenizer.encode_plus(self.combined['text'][idx], max_length=512, padding='max_length',
                                          truncation=True)
        for k, v in data.items():
            data[k] = torch.Tensor(v).long()

        data['labels'] = torch.Tensor([1] if self.combined['label'][idx] else [0]).long()

        return data
