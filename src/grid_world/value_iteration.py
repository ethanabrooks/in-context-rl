import math

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from grid_world.grid_world import GridWorld


class ValueIteration(GridWorld):
    def __init__(
        self, alpha: float, atol: float = 0.02, gamma: float = 0.99, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.atol = atol
        self.gamma = gamma

    def check_V(self, V: torch.Tensor):
        B = self.n_tasks
        N = self.n_states
        assert [*V.shape] == [B, N]

    def value_iteration(
        self,
        n_policies: int,
        n_rounds: int,
    ):
        B = self.n_tasks
        N = self.grid_size**2 + 1
        A = len(self.deltas)
        states = torch.tensor(
            [[i, j] for i in range(self.grid_size) for j in range(self.grid_size)]
        )
        alpha = torch.ones(4)
        Pi = (
            torch.distributions.Dirichlet(alpha)
            .sample((n_policies, N))
            .tile(math.ceil(B / n_policies), 1, 1)[:B]
        )
        self.check_pi(Pi)

        # Compute next states for each action and state for each batch (goal)
        next_states = states[:, None] + self.deltas[None, :]
        next_states = torch.clamp(next_states, 0, self.grid_size - 1)
        S_ = (
            next_states[..., 0] * self.grid_size + next_states[..., 1]
        )  # Convert to indices

        # Determine if next_state is the goal for each batch (goal)
        is_goal = (self.goals[:, None] == states[None]).all(-1)

        # Modify transition to go to absorbing state if the next state is a goal
        absorbing_state_idx = N - 1
        S_ = S_[None].tile(B, 1, 1)
        S_[is_goal[..., None].expand_as(S_)] = absorbing_state_idx

        # Insert row for absorbing state
        padding = (0, 0, 0, 1)  # left 0, right 0, top 0, bottom 1
        S_ = F.pad(S_, padding, value=absorbing_state_idx)
        R = is_goal.float()[..., None].tile(1, 1, A)
        R = F.pad(R, padding, value=0)  # Insert row for absorbing state

        V = None
        for _ in range(n_rounds):
            V = self.policy_evaluation(Pi, V)
            # self.visualize_values(V)
            Pi = self.policy_improvement(Pi, V)
            # self.visualize_policy(Pi)
            yield V, Pi

    def policy_evaluation(self, Pi: torch.Tensor, V: torch.Tensor = None):
        self.check_pi(Pi)

        B = self.n_tasks
        N = self.n_states
        A = len(self.deltas)
        assert [*Pi.shape] == [B, N, A]

        # Compute the policy conditioned transition function
        Pi_ = Pi.view(B * N, 1, A)
        T_ = self.T.view(B * N, A, N)
        T_Pi = torch.bmm(Pi_, T_)
        T_Pi = T_Pi.view(B, N, N)

        # Initialize V_0
        if V is None:
            V = torch.zeros((B, N), dtype=torch.float32)
        self.check_V(V)
        ER = (Pi * self.R).sum(-1)
        EV = (T_Pi * V[:, None]).sum(-1)
        V = ER + self.gamma * EV
        return V

    def policy_improvement(self, Pi: torch.Tensor, V: torch.Tensor):
        self.check_pi(Pi)
        self.check_V(V)
        Q = self.R + self.gamma * (self.T * V[:, None, None]).sum(-1)
        new_Pi = torch.zeros_like(Pi)
        new_Pi.scatter_(-1, Q.argmax(dim=-1, keepdim=True), 1.0)
        return (1 - self.alpha) * Pi + (self.alpha) * new_Pi

    def visualize_values(self, V: torch.Tensor, task_idx: int = 0):
        global_min = V[task_idx].min().item()
        global_max = V[task_idx].max().item()

        values = V[task_idx, :-1].reshape((self.grid_size, self.grid_size))
        fig, ax = plt.subplots()
        im = ax.imshow(
            values,
            cmap="hot",
            interpolation="nearest",
            vmin=global_min,
            vmax=global_max,
        )

        # Add colorbar to each subplot
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(f"values{task_idx}.png")
