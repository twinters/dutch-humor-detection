import json
import random

import gensim
import pandas as pd
import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):

    def __init__(self, original: str, replaced: str, embeddings_path: str):
        print("loading word embeddings from {}".format(embeddings_path))
        self.embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path)
        print("loaded word embeddings")

        with open(original) as fp:
            self.df = pd.DataFrame(json.load(fp), columns=["text"])
            self.df['label'] = 0

        with open(replaced) as fp:
            self.df2 = pd.DataFrame(json.load(fp), columns=['text'])
            self.df2['label'] = 1

            self.combined = self.df.append(self.df2, ignore_index=True)
            self.combined = self.combined.sample(frac=1.0)  # Shuffle

    def get_embeddings(self):
        return self.embeddings.wv.vectors

    def __len__(self):
        return len(self.combined)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = {}

        data['input_ids'] = [(self.embeddings.vocab[w].index if w in self.embeddings.vocab else 0) for w in
                             self.combined['text'][idx].lower().replace(".", "").replace("\\n", " ").split(" ")]
        data['label'] = torch.Tensor([1] if self.combined['label'][idx] else [0]).long()

        data['input_ids'] = data['input_ids'][:128]
        while len(data['input_ids']) < 128:
            data['input_ids'].append(0)

        data['input_ids'] = torch.Tensor(data['input_ids']).long()

        return data


class RankSequenceDataset(Dataset):

    def __init__(self, original: str, replaced: str, embeddings_path: str):
        print("loading word embeddings from {}".format(embeddings_path))
        self.embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path)
        print("loaded word embeddings")

        with open(original) as fp:
            self.df = pd.DataFrame(json.load(fp), columns=["text"])

        with open(replaced) as fp:
            self.df['scrambled'] = pd.DataFrame(json.load(fp), columns=['scrambled'])['scrambled']

    def get_embeddings(self):
        return self.embeddings.wv.vectors

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        random_position = random.random() > 0.5

        data = {}

        data['a' if random_position else 'b'] = [(self.embeddings.vocab[w].index if w in self.embeddings.vocab else 0)
                                                 for w in
                                                 self.df['text'][idx].lower().replace(".", "").replace("\\n",
                                                                                                       " ").split(" ")]
        data['b' if random_position else 'a'] = [(self.embeddings.vocab[w].index if w in self.embeddings.vocab else 0)
                                                 for w in
                                                 self.df['scrambled'][idx].lower().replace(".", "").replace("\\n",
                                                                                                       " ").split(" ")]
        data['label'] = torch.Tensor([1] if random_position else [0]).long()

        for x in ['a', 'b']:
            data[x] = data[x][:128]
            while len(data[x]) < 128:
                data[x].append(0)

            data[x] = torch.Tensor(data[x]).long()

        return data
