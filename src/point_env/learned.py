import os
from collections import defaultdict
from dataclasses import asdict, astuple
from functools import lru_cache
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gym.wrappers import TimeLimit
from matplotlib.axes import Axes
from matplotlib import patches
from omegaconf import OmegaConf

import data
import replay_buffer
from data import Step
from encoder import ContiguousEncoder
from plot import plot_eval_metrics
from point_env.env import PointEnv
from pretty import render_eval_metrics
from replay_buffer import valid_checksum


def get_history_trajectories(data: np.ndarray, history_index: int):
    # Extract where done is True to split the histories
    done_indices, _ = np.where(data["done"])
    done_indices += 1  # +1 to include the last step in the split
    done_indices = np.pad(done_indices, (1, 0), constant_values=0)
    start_idx = done_indices[history_index]
    end_idx = done_indices[history_index + 1]

    history_data = data[start_idx:end_idx]

    # Extract trajectories within the history using done_mdp
    start_traj_idx = 0
    for i, entry in enumerate(history_data):
        if entry["done_mdp"]:
            yield (history_data[start_traj_idx : i + 1])
            start_traj_idx = i + 1


def plot_trajectory(goal: np.ndarray, states: np.ndarray, ax: Optional[Axes] = None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    plt.scatter(*goal, c="r", marker="*", label="Goal", s=200)
    plt.plot(states[:, 0], states[:, 1], "-o", label="States Visited")

    # Mark start (now the second state) and end of the trajectory
    ax.scatter(states[0, 0], states[0, 1], c="g", marker="^", s=150, label="Start")
    ax.scatter(states[-1, 0], states[-1, 1], c="r", marker="v", s=150, label="End")

    # Add a circle with radius of 1 centered at (0,0) with a dashed line
    circle = patches.Circle((0, 0), 1, fill=False, linestyle="--", edgecolor="gray")
    ax.add_patch(circle)

    # Fixing the axis limits
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect(
        "equal", "box"
    )  # this ensures the circle is indeed circular in the plot

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    return fig


def visualize_history(data: np.ndarray, history_index: int):
    trajectories = list(get_history_trajectories(data, history_index))

    # Extract first, middle, and last trajectories
    first_traj = trajectories[0]
    middle_traj = trajectories[len(trajectories) // 2]
    last_traj = trajectories[-1]

    def _plot_trajectory(trajectory: np.ndarray):
        goal = trajectory[0]["task"]

        # Consider the second state as the start, and slice the states accordingly
        states = trajectory["state"]
        plot_trajectory(goal, states, ax=plt.gca())

    # Plot the trajectories
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    _plot_trajectory(first_traj)
    plt.title("First Trajectory")
    plt.xlim(-1.5, 1.5)  # Setting the same limits for all the plots
    plt.ylim(-1.5, 1.5)

    plt.subplot(1, 3, 2)
    _plot_trajectory(middle_traj)
    plt.title("Middle Trajectory")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    plt.subplot(1, 3, 3)
    _plot_trajectory(last_traj)
    plt.title("Last Trajectory")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.savefig("history.png")


def expand_as(x: np.ndarray, y: np.ndarray):
    while x.ndim < y.ndim:
        x = x[..., None]
    return np.broadcast_to(x, y.shape)


class Data(data.Data):
    def __init__(
        self,
        decimals: float,
        episode_length: int,
        episodes_per_rollout: int,
        expert_distillation: bool,
        include_task: bool,
        steps_per_context: int,
    ):
        self._episode_length = episode_length
        self._episodes_per_rollout = episodes_per_rollout
        self._include_task = include_task
        self.steps_per_context = steps_per_context
        components = self.get_data()

        def ends_to_starts(ends: np.ndarray):
            return np.concatenate([[0], ends[:-1] + 1])

        if expert_distillation:
            episode_end, _ = components["done_mdp"].nonzero()
            episode_start = ends_to_starts(episode_end)
            history_end, _ = components["done"].nonzero()
            history_start = ends_to_starts(history_end)
            episode_starts_per_history = np.split(
                episode_start, np.searchsorted(episode_end, history_start[1:])
            )
            n = 100

            def generate_starts():
                for starts in episode_starts_per_history:
                    start, *_ = starts[-n:]
                    yield start

            starts = list(generate_starts())
            ends = np.append(history_start[1:], -1)
            components = np.concatenate(
                [components[start:end] for start, end in zip(starts, ends)], axis=0
            )

            # Step 1: Compute a mask for each "done" value

        # Fix bug where first step is omitted
        episode_end, _ = components["done_mdp"].nonzero()
        episode_start = ends_to_starts(episode_end)
        states = components["state"]
        starts = episode_start[:-1]
        states[starts] = np.zeros_like(states[starts])

        def make_mask(component: np.ndarray):
            mask = expand_as(~components["done_mdp"], component)
            mask = np.roll(mask, shift=1, axis=1)
            return mask

        self.tasks: np.ndarray = components["task"].round(decimals)
        if not include_task:
            self.tasks = np.zeros_like(self.tasks)
        actions = components["actions"]
        actions = PointEnv.clip_action(actions)
        self.actions: np.ndarray = actions.round(decimals)
        self.observations: np.ndarray = components["state"].round(decimals)
        self.rewards: np.ndarray = components["rewards"].round(decimals)
        self.done_mdp: np.ndarray = components["done_mdp"]
        self.ends: np.ndarray = components["done"].flatten()
        self.steps_per_row: np.ndarray = self.ends.sum()
        self.steps_per_row: np.ndarray = self.steps_per_row.astype(int)

        masks = Step(
            tasks=make_mask(self.tasks),
            observations=make_mask(self.observations),
            actions=np.ones_like(self.actions),
            rewards=np.ones_like(self.rewards),
        )
        data = Step(
            tasks=self.tasks,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
        )
        self.unpadded_mask = np.concatenate(astuple(masks), axis=-1)
        self.unpadded_data = np.concatenate(astuple(data), axis=-1)
        self.unpadded_ends = self.ends.copy()
        encoder_input = self.unpadded_data.flatten()
        encoder_input = np.append(encoder_input, self.pad_value)
        encoder_input = torch.from_numpy(encoder_input)
        self._encoder = ContiguousEncoder(encoder_input, decimals)
        padding = ((self.steps_per_context, 0), (0, 0))
        mask = np.pad(self.unpadded_mask, padding, constant_values=False)
        data = np.pad(self.unpadded_data, padding, constant_values=self.pad_value)
        ends = np.pad(self.unpadded_ends, padding[0], constant_values=0)
        self.mask = torch.Tensor(mask).bool().cuda()
        self.data = torch.Tensor(data).cuda()
        self.ends = torch.Tensor(ends).cuda()

    def __getitem__(self, idx):
        idxs = slice(idx, idx + self.steps_per_context)

        # mask
        episode_mask = self.mask[idxs]
        history_ends = self.ends[idxs]
        history_counts = history_ends.cumsum(0)
        history_mask: torch.Tensor = history_counts == history_counts[-1]
        history_mask = history_mask[..., None].expand_as(episode_mask)
        mask = history_mask & episode_mask

        return self.data[idxs].flatten(), mask.flatten()

    def __len__(self):
        return len(self.unpadded_data)

    @property
    def context_size(self):
        return self.steps_per_context * self.step_dim

    @property
    def dims(self):
        _, goal_dim = self.tasks.shape
        _, obs_dim = self.observations.shape
        _, act_dim = self.actions.shape
        return Step(tasks=goal_dim, observations=obs_dim, actions=act_dim, rewards=1)

    @property
    def encoder(self) -> ContiguousEncoder:
        return self._encoder

    @property
    def episode_length(self):
        return self._episode_length

    @property
    def episodes_per_rollout(self):
        return self._episodes_per_rollout

    @property
    def eval_metric_name(self) -> str:
        return "regret"

    @property
    def include_task(self):
        return self._include_task

    @property
    @lru_cache
    def max_regret(self):
        return self.episode_length

    @property
    @lru_cache
    def episode_rewards(self):
        # Get indices where done_mdp is True
        end_indices, _ = np.where(self.done_mdp)
        end_indices += 1  # +1 to include the last step in the split

        # Split the rewards array at the end of each episode
        split_rewards = np.split(self.rewards, end_indices)
        return [r for r in split_rewards if r.size > 0]

    @property
    @lru_cache
    def n_tokens(self):
        return self.encoder.bins.numel() - 1

    @property
    @lru_cache
    def pad_value(self):
        return self.unpadded_data.max() + 1

    def build_env(
        self,
        seed: int,
        use_heldout_tasks: bool,
        include_optimal: bool = True,
        max_episode_steps: Optional[int] = None,
    ):
        if include_optimal:
            returns = [
                np.sum(episode_rewards) for episode_rewards in self.episode_rewards
            ]
            # Get the index of the episode with the best return
            best_index = np.argmax(returns)
            optimal = self.episode_rewards[best_index].flatten()
        else:
            optimal = None
        env = PointEnv(
            goal_sampler="circle", optimal=optimal, seed=seed, test=use_heldout_tasks
        )
        if max_episode_steps is None:
            max_episode_steps = self.episode_length
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    def cat_sequence(self, step: Step[torch.Tensor]):
        data = torch.cat(astuple(step), dim=-1)
        n_data, *_ = data.shape
        return data.long().view(n_data, -1)

    def get_data(self):
        data_dir = Path(os.getenv("DATA_DIR"))
        assert data_dir.exists()

        artifacts = OmegaConf.load("artifacts.yml")
        load_path = data_dir / replay_buffer.DIR / artifacts.point_env
        assert valid_checksum(data_dir, load_path)

        return replay_buffer.load(load_path)

    def split_sequence(self, sequence: torch.Tensor):
        n_batch, _ = sequence.shape
        sequence = sequence.reshape(n_batch, -1, self.step_dim)
        dims = astuple(self.dims)
        components = sequence.split(dims, dim=-1)
        components = Step(*components)
        return components

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
        preds = logits.argmax(-1)
        preds = self.encoder.decode(preds)
        preds = torch.cat([prefix, preds], dim=1)
        tgts = sequence
        split_preds = self.split_sequence(preds)
        split_tgts = self.split_sequence(tgts)
        split_masks = self.split_sequence(mask)

        acc = {}
        acc["(total) accuracy"] = (preds == tgts)[mask.bool()]
        iterator = list(
            zip(
                asdict(split_preds).items(),
                asdict(split_tgts).items(),
                asdict(split_masks).items(),
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
                _, component_seq_len, *_ = pred.shape

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

    def plot_rollout(
        self,
        task: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ):
        return plot_trajectory(task, states)

    def plot_eval_metrics(self, df: pd.DataFrame) -> list[str]:
        return plot_eval_metrics(
            df, name=self.eval_metric_name, ymin=0, ymax=self.max_regret
        )

    def render_eval_metrics(self, *metric: float) -> list[str]:
        return render_eval_metrics(*metric, max_num=self.max_regret)


# whitelist
visualize_history
