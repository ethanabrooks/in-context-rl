from abc import ABC, abstractmethod
from functools import lru_cache

import torch
from torch.utils.data import Dataset


class Data(Dataset, ABC):
    @property
    def n_tokens(self):
        return 1 + self.data.max().round().long().item()

    @property
    def step_dim(self):
        return sum(self._dims)

    @abstractmethod
    def cat_sequence(self, goals, observations, actions, rewards):
        raise NotImplementedError

    @abstractmethod
    def get_metrics(
        self, logits: torch.Tensor, graphs_per_component: int, sequence: torch.Tensor
    ):
        raise NotImplementedError

    @abstractmethod
    def split_sequence(self, sequence: torch.Tensor):
        raise NotImplementedError

    @lru_cache
    def weights(self, shape, **kwargs):
        weights = torch.ones(shape)
        sequence = self.split_sequence(weights)
        for k, v in kwargs.items():
            assert k in sequence, f"Invalid key {k}"
            sequence[k] *= v
        return self.cat_sequence(**sequence).cuda()


def unwrap(dataset: Dataset):
    if isinstance(dataset, Data):
        return dataset
    return unwrap(dataset.dataset)
