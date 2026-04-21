from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset


class PseudoShuffledIterableDataset(IterableDataset):
    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []
        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
                yield shuffle_buffer.pop(idx)
        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class BearingRULIterableDataset(IterableDataset, ShuffleMixin):
    def __init__(self, dataset_path: str | Path, mode: str = 'train', top_k: int = 5):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.mode = mode
        self.top_k = int(top_k)
        if not self.dataset_path.exists():
            raise FileNotFoundError(self.dataset_path)
        self.df = pd.read_parquet(self.dataset_path)

    def __iter__(self):
        while True:
            for _, row in self.df.iterrows():
                item = {
                    'x': np.asarray(row['x'], dtype=np.float32),
                    'y': np.float32(row['y_rul_norm']),
                    'y_steps': np.float32(row['y_rul_steps']),
                    'num_files': np.float32(row['num_files']),
                    'query_features': np.asarray(row['query_features'], dtype=np.float32),
                    'indices': np.asarray(row['indices'][:self.top_k], dtype=np.int64),
                    'distances': np.asarray(row['distances'][:self.top_k], dtype=np.float32),
                    'bearing_id': row['bearing_id'],
                    'end_idx': np.int64(row['end_idx']),
                }
                yield item
            if self.mode != 'train':
                break


class BearingRULEvalDataset(Dataset):
    def __init__(self, dataset_path: str | Path, top_k: int = 5):
        self.dataset_path = Path(dataset_path)
        self.top_k = int(top_k)
        if not self.dataset_path.exists():
            raise FileNotFoundError(self.dataset_path)
        self.df = pd.read_parquet(self.dataset_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return {
            'x': torch.tensor(np.asarray(row['x'], dtype=np.float32)),
            'y': torch.tensor(np.float32(row['y_rul_norm'])),
            'y_steps': torch.tensor(np.float32(row['y_rul_steps'])),
            'num_files': torch.tensor(np.float32(row['num_files'])),
            'query_features': torch.tensor(np.asarray(row['query_features'], dtype=np.float32)),
            'indices': torch.tensor(np.asarray(row['indices'][:self.top_k], dtype=np.int64)),
            'distances': torch.tensor(np.asarray(row['distances'][:self.top_k], dtype=np.float32)),
            'bearing_id': row['bearing_id'],
            'end_idx': int(row['end_idx']),
        }
