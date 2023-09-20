from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import evaluators.ad
from evaluators.ad import StepResult, clamp


@dataclass
class Evaluator(evaluators.ad.Evaluator):
    n_actions: int

    def make_rollout(self, *args, **kwargs):
        return Rollout(*args, **kwargs, n_actions=self.n_actions)


@dataclass
class PlanningRollout(evaluators.ad.Rollout):
    history: torch.Tensor
    t: int
    episode_t: np.ndarray

    def __post_init__(self):
        pass

    @property
    def episode_length(self):
        return 1 + self.dataset.episode_length - self.episode_t.max()

    @property
    def episodes_per_rollout(self):
        return 1

    def index(self, t: int):
        return super().index(t + self.t)

    def init_history(self):
        T = self.dataset.episode_length * self.dataset.episodes_per_rollout
        required_width = self.index(T)
        _, L = self.history.shape
        pad = required_width - L
        if pad > 0:
            return F.pad(self.history, (0, pad), value=self.dataset.pad_value)
        else:
            return self.history

    def reset(self, *_, **__):
        pass  # No reset. Only one episode.

    def step(self, action: torch.Tensor, history: torch.Tensor, t: int) -> torch.Tensor:
        O = self.dataset.dims.observations

        [reward] = self.predict_many(history, t, *self.idxs.rewards).T
        assert [*reward.shape] == [self.n_rollouts]

        i, j = self.index(t) + self.idxs.tasks
        history[:, i:j] = self.task

        observation = self.predict_many(history, t + 1, *self.idxs.observations)
        assert [*observation.shape] == [self.n_rollouts, O]

        done = t + 1 == self.episode_length
        done = np.full(reward.shape, fill_value=done, dtype=bool)

        return StepResult(
            reward=reward.cpu().numpy(),
            observation=observation.cpu().numpy(),
            done=done,
            info=[{} for _ in range(self.n_rollouts)],
        )


@dataclass
class Rollout(evaluators.ad.Rollout):
    n_actions: int

    def get_action(self, history: torch.Tensor, t: int, episode_t) -> torch.Tensor:
        planner = PlanningRollout(
            dataset=self.dataset,
            envs=self.envs,
            episode_t=episode_t,
            gamma=self.gamma,
            history=history.repeat(self.n_actions, 1),
            n_rollouts=self.n_rollouts * self.n_actions,
            net=self.net,
            t=t,
            task=self.task.repeat(self.n_actions, 1),
        )
        rollouts = list(planner.rollout())
        rollouts = pd.DataFrame.from_records(rollouts)
        rollouts["row"] = rollouts.n // self.n_actions
        idx = rollouts.groupby("row").metric.idxmax()
        histories = rollouts.loc[idx].history.tolist()
        histories = torch.stack(histories).cuda()
        i, j = self.index(t) + self.idxs.actions
        actions = histories[:, i:j]
        return clamp(actions, self.envs.action_space)
