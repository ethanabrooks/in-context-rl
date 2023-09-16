from abc import ABC, abstractmethod
from functools import lru_cache

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from envs.base import Env


class Data(Dataset, ABC):
    @property
    @abstractmethod
    def episode_length(self) -> int:
        pass

    @property
    @abstractmethod
    def include_goal(self) -> bool:
        pass

    @property
    @abstractmethod
    def return_range(self) -> tuple[float, float]:
        pass

    @property
    @lru_cache
    def n_tokens(self):
        return 1 + self.data.max().round().long().item()

    @property
    @lru_cache
    def pad_value(self):
        return self.data.max().round().long().item()

    @property
    @abstractmethod
    def episodes_per_rollout(self):
        pass

    @property
    def step_dim(self):
        return sum(self._dims)

    @abstractmethod
    def build_env(self) -> Env:
        pass

    @abstractmethod
    def cat_sequence(self, goals, observations, actions, rewards) -> torch.Tensor:
        pass

    @abstractmethod
    def get_metrics(
        self, logits: torch.Tensor, graphs_per_component: int, sequence: torch.Tensor
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        pass

    @abstractmethod
    def split_sequence(self, sequence: torch.Tensor):
        pass

    @lru_cache
    def weights(self, shape, **kwargs):
        weights = torch.ones(shape)
        sequence = self.split_sequence(weights)
        for k, v in kwargs.items():
            assert k in sequence, f"Invalid key {k}"
            sequence[k] *= v
        return self.cat_sequence(**sequence).cuda()


def plot_accuracy(
    *accuracies: float,
    name: str,
    ymin: float,
    ymax: float,
):
    fig, ax = plt.subplots()
    x = list(range(len(accuracies)))
    ax.bar(x, accuracies)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("context step")
    ax.set_ylabel(f"{name} accuracy")
    ax.grid(True)
    return fig
