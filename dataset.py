from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.utils import shuffle
import pandas as pd
import numpy as np


MAX_TOKENS_PER_COLUMN = 200  # 2 of those for CLS and SEP
MAX_COLUMNS = 20
MAX_TOKENS = 200  # per table

np.random.seed(42)


class Tables(Dataset):

    def __collate_fn(samples):
        tokens = torch.nn.utils.rnn.pad_sequence(
            list(map(lambda x: x["input_ids"], samples)))
        labels = torch.cat(list(map(lambda x: x["labels"], samples)))

        return {"input_ids": tokens.T, "labels": labels}

    def __init__(self, paths, tokenizer, labels_encoder, use_rand=False):
        self.paths = paths
        self.use_rand = use_rand
        self.tokenizer = tokenizer
        self.labels_encoder = labels_encoder

    def __len__(self):
        return len(self.paths) - 1

    def __getitem__(self, idx):
        df = pd.read_csv(self.paths[idx], sep="|")

        tokens = []
        with open(self.paths[idx]) as file:
            labels = file.readline().lower().rstrip('\n').split("|")

        assert len(labels) == len(df.columns)

        columns = df.columns[:MAX_COLUMNS]
        labels = labels[:MAX_COLUMNS]

        if self.use_rand:
            cols_order, labels = zip(
                *np.random.permutation(list(zip(columns, labels))))
            df = df[list(cols_order)]
            df = shuffle(df)

        tokens_per_column = min(
            MAX_TOKENS // len(labels), MAX_TOKENS_PER_COLUMN)

        for label, _ in zip(df.columns, range(MAX_COLUMNS)):
            str_repr_of_column = df[label].astype(str).str.cat(sep=" ")
            tokens += self.tokenizer(str_repr_of_column, truncation=True,
                                     max_length=tokens_per_column).input_ids

        labels = self.labels_encoder.transform(labels)[:MAX_COLUMNS]

        return {'input_ids': torch.tensor(tokens),
                'labels': torch.tensor(labels)}

    def create_dataloader(self, batch_size=40, num_workers=1, shuffle=False):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers,
                          shuffle=shuffle, collate_fn=Tables.__collate_fn)


class Columns(Dataset):

    def __init__(self, path, tokenizer, labels_encoder):
        self.df = pd.read_csv(path, sep="<")
        self.tokenizer = tokenizer
        self.labels_encoder = labels_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.labels_encoder.transform(
            [str(self.df["label"].iloc[idx])])[0]
        str_repr_of_column = str(self.df["column"].iloc[idx])
        tokens = self.tokenizer(str_repr_of_column, truncation=True, padding="max_length",
                                max_length=30).input_ids

        return {'input_ids': torch.tensor(tokens),
                'labels': torch.tensor(label)}

    def create_dataloader(self, batch_size=40, num_workers=1, shuffle=False):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers,
                          shuffle=shuffle)
