from collections import defaultdict

import torch
import torch.nn.functional as F

import data.base
from envs.grid_world_env import Env
from envs.value_iteration import ValueIteration
from pretty import console


def expand_as(x: torch.Tensor, y: torch.Tensor):
    while x.dim() < y.dim():
        x = x[..., None]
    return x.expand_as(y)


class Data(data.base.Data):
    def __init__(
        self,
        alpha: float,
        grid_size: int,
        grid_world_args: dict,
        include_goal: bool,
        mask_nonactions: bool,
        n_data: int,
        n_episodes: int,
        steps_per_context: int,
        value_iteration_args: dict,
    ):
        episode_length = 1 + grid_size**2
        grid_world = ValueIteration(
            alpha=alpha,
            episode_length=episode_length,
            **grid_world_args,
            grid_size=grid_size,
            n_tasks=n_data,
        )
        self._include_goal = include_goal
        self.episode_length = grid_world.episode_length
        self.grid_size = grid_size
        self.n_data = n_data
        self.n_episodes = n_episodes
        self.steps_per_context = steps_per_context
        self.n_rounds = round((2 * grid_size - 1) / alpha)

        data = list(self.collect_data(grid_world, **value_iteration_args))
        components = zip(*data)
        components = [torch.cat(c, dim=1) for c in components]
        (
            self.goals,
            self.observations,
            self.actions,
            self.rewards,
            self.done,
        ) = components
        if not include_goal:
            self.goals = torch.zeros_like(self.goals)
        components = [self.goals, self.observations, self.actions, self.rewards]
        masks = [
            expand_as(~self.done, c).roll(dims=[1], shifts=[1])
            for c in [self.goals, self.observations]
        ] + [torch.ones_like(c) for c in [self.actions, self.rewards]]
        if mask_nonactions:
            goals_mask, obs_mask, actions_mask, rewards_mask = masks
            masks = [
                torch.zeros_like(goals_mask),
                torch.zeros_like(obs_mask),
                actions_mask,
                torch.zeros_like(rewards_mask),
            ]

        mask = self.cat_sequence(*masks)
        data = self.cat_sequence(*components)
        sequence = self.split_sequence(data)
        for name, component in sequence.items():
            assert (getattr(self, name) == component).all()

        data = data.reshape(n_data, -1, self.step_dim)
        mask = mask.reshape(n_data, -1, self.step_dim)
        _, self.steps_per_row, _ = data.shape
        self.unpadded_data = data
        self.unpadded_mask = mask
        assert [*self.unpadded_data.shape] == [
            n_data,
            self.steps_per_row,
            self.step_dim,
        ]
        pad_value = self.unpadded_data.max().item() + 1
        self.data = F.pad(data, (0, 0, steps_per_context, 0), value=pad_value).cuda()
        self.mask = F.pad(mask, (0, 0, steps_per_context, 0), value=0).cuda()

    def __getitem__(self, idx):
        i, j = self.index_1d_to_2d(idx)
        jj = slice(j, j + self.steps_per_context)
        return self.data[i, jj].view(-1), self.mask[i, jj].view(-1)

    def __len__(self):
        return self.n_data * self.steps_per_row

    @property
    def _dims(self):
        _, _, goal_dim = self.goals.shape
        _, _, obs_dim = self.observations.shape
        return [goal_dim, obs_dim, 1, 1]

    @property
    def include_goal(self):
        return self._include_goal

    @property
    def return_range(self):
        return 0.0, 1.0

    @property
    def episodes_per_rollout(self):
        return self.n_rounds * self.n_episodes

    def build_env(self):
        return Env(grid_size=self.grid_size, episode_length=self.episode_length)

    def cat_sequence(self, goals, observations, actions, rewards):
        data = torch.cat(
            [goals, observations, actions[..., None], rewards[..., None]],
            dim=-1,
        )
        n_data, _, _ = data.shape
        return data.long().reshape(n_data, -1).contiguous()

    def collect_data(self, grid_world: ValueIteration, **kwargs):
        console.log("Value iteration...")
        for t, (V, Pi) in enumerate(
            (grid_world.value_iteration(**kwargs, n_rounds=self.n_rounds))
        ):
            g, s, a, r, d = grid_world.get_trajectories(
                Pi=Pi, n_episodes=self.n_episodes
            )
            console.log(
                f"Round: {t}. Reward: {r.sum(-1).mean().item():.2f}. Value: {V.mean().item():.2f}."
            )
            yield g, s, a, r, d

    def index_1d_to_2d(self, index):
        row = index // self.steps_per_row
        col = index % self.steps_per_row
        return (row, col)

    def split_sequence(self, sequence: torch.Tensor):
        n_batch, _ = sequence.shape
        sequence = sequence.reshape(n_batch, -1, self.step_dim)
        goals, observations, actions, rewards = sequence.split(self._dims, dim=-1)
        actions = actions.squeeze(-1)
        rewards = rewards.squeeze(-1)
        return dict(
            goals=goals, observations=observations, actions=actions, rewards=rewards
        )

    def get_metrics(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
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
        split_masks = self.split_sequence(mask)

        acc = {}
        acc["(total) accuracy"] = (preds == tgts)[mask.bool()]
        iterator = list(
            zip(
                split_preds.items(),
                split_tgts.items(),
                split_masks.items(),
            )
        )
        for (name, pred), (name2, tgt), (name3, mask) in iterator:
            assert name == name2 == name3
            total_accuracy = (pred == tgt)[mask.bool()]
            acc[f"(total) {name} accuracy"] = total_accuracy

        table = defaultdict(list)
        for i in range(seq_len):
            for (name, pred), (name2, tgt), (name3, mask) in iterator:
                assert name == name2 == name3
                _, component_seq_len, *_ = pred[mask].shape

                if i >= component_seq_len:
                    continue

                def get_chunk(x):
                    if x.ndim == 2:
                        x = x[..., None]
                    return x[:, i : i + 1]

                pred_chunk = get_chunk(pred)
                tgt_chunk = get_chunk(tgt)
                mask_chunk = get_chunk(mask)
                if mask_chunk.sum() > 0:
                    accuracy = (pred_chunk == tgt_chunk)[mask_chunk.bool()]
                    table[f"{name} accuracy"].append(accuracy.float().mean().item())

        log = {k: v.float().mean().item() for k, v in acc.items()}
        return log, table
