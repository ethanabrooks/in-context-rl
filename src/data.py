from functools import lru_cache
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import get_trajectories


class RLData(Dataset):
    def __init__(
        self,
        grid_size: int,
        n_data: int,
    ):
        episode_length = 1 + grid_size * 2
        self.observations, self.actions, self.rewards = get_trajectories(
            grid_size=grid_size,
            n_data=n_data,
            episode_length=episode_length,
            n_episodes=1,
        )
        self.data = self.cat_data(self.observations, self.actions, self.rewards)
        sequence = self.split_sequence(self.data)
        for name, component in dict(
            observations=self.observations, actions=self.actions, rewards=self.rewards
        ).items():
            assert (sequence[name] == component).all()
        self.data = self.data.cuda()
        self.mask = torch.ones_like(self.data).cuda()

    @property
    def n_tokens(self):
        return 1 + self.data.max().round().long().item()

    @property
    def _observation_dim(self):
        _, _, observation_dim = self.observations.shape
        return observation_dim

    @property
    def step_dim(self):
        return self._observation_dim + 1 + 1

    def cat_data(self, observations, actions, rewards):
        data = torch.cat(
            [observations, actions[..., None], rewards[..., None]],
            dim=-1,
        )
        n_data, _, _ = data.shape
        return data.long().reshape(n_data, -1).contiguous()

    def split_sequence(self, sequence: torch.Tensor):
        n_batch, _ = sequence.shape
        sequence = sequence.reshape(n_batch, -1, self.step_dim)

        observations = sequence[:, :, : self._observation_dim]
        actions = sequence[:, :, self._observation_dim]
        rewards = sequence[:, :, -1]
        return dict(observations=observations, actions=actions, rewards=rewards)

    @lru_cache
    def weights(self, shape, **kwargs):
        weights = torch.ones(shape)
        sequence = self.split_sequence(weights)
        for k, v in kwargs.items():
            assert k in sequence, f"Invalid key {k}"
            sequence[k] *= v
        return self.cat_data(**sequence).cuda()

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]

    def __len__(self):
        return len(self.data)

    def get_metrics(
        self,
        logits: torch.Tensor,
        graphs_per_component: int,
        sequence: torch.Tensor,
    ):
        n_batch, seq_len = sequence.shape
        n_batch2, seq_len2, _ = logits.shape
        assert n_batch == n_batch2
        assert seq_len == seq_len2 + 1

        prefix = sequence[:, :1]
        preds = torch.cat([prefix, logits.argmax(-1)], dim=1)
        tgts = sequence
        split_preds = self.split_sequence(preds)
        split_tgts = self.split_sequence(tgts)

        acc = {}
        for (name, pred), (name2, tgt) in zip(
            dict(**split_preds, total=preds).items(),
            dict(**split_tgts, total=tgts).items(),
        ):
            assert name == name2
            acc[f"{name} accuracy"] = pred == tgt

        chunk_acc = {}
        for (name, pred), (name2, tgt) in zip(
            split_preds.items(),
            split_tgts.items(),
        ):
            assert name == name2
            acc[f"{name} accuracy"] = pred == tgt
            _, seq_len, *_ = pred.shape
            chunk_size = seq_len // graphs_per_component
            for i in range(graphs_per_component):
                start = i * chunk_size
                end = (i + 1) * chunk_size

                def get_chunk(x):
                    if x.ndim == 2:
                        x = x[..., None]
                    return x[:, start:end]

                pred_chunk = get_chunk(pred)
                tgt_chunk = get_chunk(tgt)
                chunk_acc[f"({i}) {name} accuracy"] = pred_chunk == tgt_chunk

        logs = dict(**acc, **chunk_acc)
        return {k: v.float().mean().item() for k, v in logs.items()}


def unwrap(dataset: Dataset):
    if isinstance(dataset, RLData):
        return dataset
    return unwrap(dataset.dataset)
