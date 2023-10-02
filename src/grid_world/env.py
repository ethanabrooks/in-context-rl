from typing import Union

import torch
from gym.spaces import Discrete, MultiDiscrete

from envs.base import Env
from grid_world.grid_world import GridWorld


class Env(GridWorld, Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_tasks=1, **kwargs)
        self.current_state = None
        self.t = None

    @property
    def action_space(self):
        return Discrete(len(self.deltas))

    @property
    def observation_space(self):
        return MultiDiscrete([self.grid_size, self.grid_size])

    @property
    def task_space(self):
        return MultiDiscrete([self.grid_size, self.grid_size])

    def get_task(self):
        [goal] = self.goals
        return goal

    def reset(self):
        self.current_state = self.reset_fn()
        [s] = self.current_state
        self.t = 0
        distance = torch.abs(self.get_task() - s).sum()
        if self.dense_reward:
            descending = list(range(-distance, 0))
            self.optimal = descending + [0.0] * (self.episode_length - distance)
        else:
            self.optimal = (
                [0.0] * distance
                + [1.0]
                + [0.0 if self.absorbing_state else 1.0]
                * (self.episode_length - distance - 1)
            )
        return s.numpy()

    def step(self, action: Union[torch.Tensor, int]):
        if isinstance(action, int):
            action = torch.tensor([action])
        action = action.reshape(1)
        self.current_state, [r], d, i = self.step_fn(self.current_state, action, self.t)
        [s] = self.current_state
        self.t += 1
        if d:
            i.update(optimal=self.optimal)
        return s.numpy(), r, d, i
