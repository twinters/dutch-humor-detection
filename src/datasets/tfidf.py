import json
import random

import numpy as np
import pandas as pd
import typing

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):

    def __init__(self, original: str, replaced: str):

        with open(original) as fp:
            self.df = pd.DataFrame(json.load(fp), columns=["text"])
            self.df['label'] = 0

        with open(replaced) as fp:
            self.df2 = pd.DataFrame(json.load(fp), columns=['text'])
            self.df2['label'] = 1

            self.combined = self.df.append(self.df2, ignore_index = True)
            self.combined = self.combined.sample(frac=1.0) # Shuffle

            self.vectorizer = TfidfVectorizer()
            self.x = self.vectorizer.fit_transform(self.combined['text'])
            self.y = self.combined['label']


    def get_embedding_size(self):
        return len(self.vectorizer.get_feature_names())

    def __len__(self):
        return len(self.combined)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        sample = {'sentence1': torch.Tensor(
            self.x[idx].toarray()[0]),
            'label': torch.Tensor([1] if self.y[idx] else [0]).long()}

        return sample


class ChooseSequenceDataset(Dataset):

    def __init__(self, original: str, replaced: str):

        with open(original) as fp:
            self.df = pd.DataFrame(json.load(fp), columns=["text"])

        with open(replaced) as fp:
            self.df['scrambled'] = pd.DataFrame(json.load(fp), columns=['scrambled'])['scrambled']

            self.vectorizer = TfidfVectorizer()
            self.x_original = self.vectorizer.fit_transform(self.df['text'])
            self.x_scrambled = self.vectorizer.transform(self.df['scrambled'])

    def _transform(self, df: pd.DataFrame):
        "Quick scrambling of the tokens."

        transformed_row: typing.List = []
        for row in df.iterrows():
            text: typing.List = row[1].text.split()
            random.shuffle(text)
            transformed_row.append(' '.join(text))

        df['scrambled'] = transformed_row

        return df

    def get_embedding_size(self):
        return len(self.vectorizer.get_feature_names())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        random_position = random.random() > 0.5

        sample = {'sentence1': torch.Tensor(
            self.x_original[idx].toarray()[0] if random_position else self.x_scrambled[idx].toarray()[0]),
            'sentence2': torch.Tensor(
                self.x_original[idx].toarray()[0] if not random_position else self.x_scrambled[idx].toarray()[0]),
            'label': torch.Tensor([1] if random_position else [0]).long()}

        return sample
