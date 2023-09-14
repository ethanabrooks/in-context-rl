import torch

import data.full_history
from tabular.grid_world import GridWorld


def expand_as(x: torch.Tensor, y: torch.Tensor):
    while x.dim() < y.dim():
        x = x[..., None]
    return x.expand_as(y)


class Data(data.full_history.Data):
    def __init__(
        self,
        grid_size: int,
        grid_world_args: dict,
        include_goal: bool,
        n_data: int,
        n_episodes: int,
        steps_per_context: int,
        value_iteration_args: dict,
    ):
        episode_length = 1 + grid_size * 2
        grid_world = GridWorld(grid_size, n_data)
        (
            self.goals,
            self.observations,
            self.actions,
            self.rewards,
            self.done,
        ) = grid_world.get_trajectories(
            episode_length=episode_length,
            n_episodes=n_episodes,
            Pi=grid_world.compute_policy_towards_goal(),
        )
        if not include_goal:
            self.goals = torch.zeros_like(self.goals)
        components = [self.goals, self.observations, self.actions, self.rewards]
        masks = [
            expand_as(~self.done, c).roll(dims=[1], shifts=[1])
            for c in [self.goals, self.observations]
        ] + [torch.ones_like(c) for c in [self.actions, self.rewards]]
        mask = self.cat_sequence(*masks)
        data = self.cat_sequence(*components)
        sequence = self.split_sequence(data)
        sequence = self.split_sequence(data)
        for name, component in sequence.items():
            assert (getattr(self, name) == component).all()
        self.data = data.cuda()
        self.mask = mask.cuda()

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]

    def __len__(self):
        return len(self.data)
