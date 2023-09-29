import functools
import importlib
from abc import ABC, abstractmethod
from dataclasses import asdict, astuple, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Generic, TypeVar, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from encoder import Encoder
from envs.base import Env
from envs.parallel.dummy_vec_env import DummyVecEnv
from envs.parallel.subproc_vec_env import SubprocVecEnv

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
    @abstractmethod
    def n_tokens(self):
        pass

    @property
    @abstractmethod
    def pad_value(self):
        pass

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

    def build_vec_envs(
        self, n_processes: int, use_heldout_tasks: bool, dummy_vec_env: bool = False
    ):
        env_fns = [
            functools.partial(
                self.build_env, seed=i, use_heldout_tasks=use_heldout_tasks
            )
            for i in range(n_processes)
        ]
        envs: SubprocVecEnv
        envs = DummyVecEnv(env_fns) if dummy_vec_env else SubprocVecEnv(env_fns)
        return envs

    @abstractmethod
    def get_metrics(
        self, logits: torch.Tensor, sequence: torch.Tensor
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        pass

    @abstractmethod
    def plot_eval_metrics(self, df: pd.DataFrame) -> list[str]:
        pass

    @abstractmethod
    def plot_rollout(
        self,
        task: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ):
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


def make(path: Union[str, Path], *args, **kwargs) -> Data:
    path = Path(path)
    name = path.stem
    name = ".".join(path.parts)
    module = importlib.import_module(name)
    data: Data = module.Data(*args, **kwargs)
    return data
