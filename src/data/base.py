from abc import ABC, abstractmethod
from dataclasses import asdict, astuple, dataclass
from functools import lru_cache
from typing import Generic, TypeVar

import pandas as pd
import torch
from torch.utils.data import Dataset

from encoder import Encoder
from envs.base import Env

T = TypeVar("T")


@dataclass(frozen=True)
class Step(Generic[T]):
    tasks: T
    observations: T
    actions: T
    rewards: T


class Data(Dataset, ABC):
    @property
    @abstractmethod
    def context_size(self) -> int:
        pass

    @property
    @abstractmethod
    def dims(self) -> Step:
        pass

    @property
    @abstractmethod
    def encoder(self) -> Encoder:
        pass

    @property
    @abstractmethod
    def episode_length(self) -> int:
        pass

    @property
    @abstractmethod
    def eval_metric_name(self) -> str:
        pass

    @property
    @abstractmethod
    def include_task(self) -> bool:
        pass

    @property
    @lru_cache
    def n_tokens(self):
        return 1 + self.encoder.encode(self.data).max().round().long().item()

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
        return sum(astuple(self.dims))

    @abstractmethod
    def build_env(self, seed: int, use_heldout_tasks: bool) -> Env:
        pass

    @abstractmethod
    def get_metrics(
        self, logits: torch.Tensor, sequence: torch.Tensor
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        pass

    @abstractmethod
    def plot_eval_metrics(self, df: pd.DataFrame) -> list[str]:
        pass

    @abstractmethod
    def render_eval_metrics(self, *metric: float) -> list[str]:
        pass

    @lru_cache
    def weights(self, shape, **kwargs):
        weights = torch.ones(shape)
        sequence = asdict(self.split_sequence(weights))
        for k, v in kwargs.items():
            assert k in sequence, f"Invalid key {k}"
            sequence[k] *= v
        return self.cat_sequence(Step(**sequence)).cuda()
