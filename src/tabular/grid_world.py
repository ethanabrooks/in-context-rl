import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm


class GridWorld:
    def __init__(self, grid_size: int, n_tasks: int):
        self.grid_size = grid_size
        self.states = torch.tensor(
            [[i, j] for i in range(grid_size) for j in range(grid_size)]
        )
        self.deltas = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])
        self.n_tasks = n_tasks
        self.goals = self.sample_goals(n_tasks)

        # Compute next states for each action and state for each batch (goal)
        next_states = self.states[:, None] + self.deltas[None, :]
        next_states = torch.clamp(next_states, 0, self.grid_size - 1)
        S_ = (
            next_states[..., 0] * self.grid_size + next_states[..., 1]
        )  # Convert to indices

        # Determine if next_state is the goal for each batch (goal)
        is_goal = (self.goals[:, None] == self.states[None]).all(-1)

        # Modify transition to go to absorbing state if the next state is a goal
        absorbing_state_idx = self.n_states - 1
        S_ = S_[None].tile(self.n_tasks, 1, 1)
        S_[is_goal[..., None].expand_as(S_)] = absorbing_state_idx

        # Insert row for absorbing state
        padding = (0, 0, 0, 1)  # left 0, right 0, top 0, bottom 1
        S_ = F.pad(S_, padding, value=absorbing_state_idx)
        self.T = F.one_hot(S_, num_classes=self.n_states).float()
        R = is_goal.float()[..., None].tile(1, 1, len(self.deltas))
        self.R = F.pad(R, padding, value=0)  # Insert row for absorbing state

    def check_pi(self, Pi: torch.Tensor):
        B = self.n_tasks
        N = self.n_states
        A = len(self.deltas)
        assert [*Pi.shape] == [B, N, A]

    def compute_policy_towards_goal(self):
        # Expand goals and states for broadcasting
        expanded_goals = self.goals[:, None, :]
        expanded_states = self.states[None, :, :]

        # Calculate the difference between each state and the goals
        diff = expanded_goals - expanded_states

        # Determine the action indices to move toward the goal for each state
        positive_x_i, positive_x_j = (diff[..., 0] > 0).nonzero(as_tuple=True)
        negative_x_i, negative_x_j = (diff[..., 0] < 0).nonzero(as_tuple=True)
        positive_y_i, positive_y_j = (diff[..., 1] > 0).nonzero(as_tuple=True)
        negative_y_i, negative_y_j = (diff[..., 1] < 0).nonzero(as_tuple=True)
        equal_i, equal_j = (diff == 0).all(-1).nonzero(as_tuple=True)

        # Initialize the actions tensor
        absorbing_state_idx = self.n_states - 1
        actions = torch.zeros(self.goals.size(0), self.n_states, 4)

        # Assign deterministic actions with vertical priority
        actions[positive_x_i, positive_x_j, 1] = 1  # Move down
        actions[negative_x_i, negative_x_j, 0] = 1  # Move up
        # Only assign horizontal actions if no vertical action has been assigned
        actions[positive_y_i, positive_y_j, 3] = (
            actions[positive_y_i, positive_y_j].sum(-1) == 0
        ).float()  # Move right
        actions[negative_y_i, negative_y_j, 2] = (
            actions[negative_y_i, negative_y_j].sum(-1) == 0
        ).float()  # Move left
        actions[
            :, absorbing_state_idx, 0
        ] = 1  # Arbitrary action, since it doesn't matter
        actions[equal_i, equal_j, 0] = 1  # Arbitrary action, since it doesn't matter

        return actions

    def get_trajectories(
        self,
        episode_length: int,
        Pi: torch.Tensor,
        n_episodes: int = 1,
    ):
        B = self.n_tasks
        N = self.n_states
        A = len(self.deltas)
        assert [*Pi.shape] == [B, N, A]

        trajectory_length = episode_length * n_episodes
        states = torch.zeros((B, trajectory_length, 2), dtype=torch.int)
        actions = torch.zeros((B, trajectory_length), dtype=torch.int)
        rewards = torch.zeros((B, trajectory_length))
        done = torch.zeros((B, trajectory_length), dtype=torch.bool)
        current_states = torch.randint(0, self.grid_size, (self.n_tasks, 2))

        for t in tqdm(range(trajectory_length), desc="Sampling trajectories"):
            if (t) % episode_length == 0:
                current_states = torch.randint(0, self.grid_size, (self.n_tasks, 2))
            if (t + 1) % episode_length == 0:
                done[:, t] = True
            # Convert current current_states to indices
            current_state_indices = (
                current_states[:, 0] * self.grid_size + current_states[:, 1]
            )

            # Sample actions from the policy
            A = (
                torch.multinomial(Pi[torch.arange(B), current_state_indices], 1)
                .squeeze(1)
                .long()
            )

            # Compute next state indices
            next_state_indices = torch.argmax(
                self.T[torch.arange(B), current_state_indices, A], dim=1
            )

            # Convert next state indices to coordinates
            next_states = torch.stack(
                (
                    next_state_indices // self.grid_size,
                    next_state_indices % self.grid_size,
                ),
                dim=1,
            )
            R = self.R[torch.arange(B), current_state_indices, A]

            next_states_, R_, _, _ = self.step_fn(current_states, A)
            if not (next_states == next_states_).all():
                breakpoint()
            if not (R == R_).all():
                breakpoint()

            # Store the current current_states and rewards
            states[:, t] = current_states
            actions[:, t] = A
            rewards[:, t] = R

            # Update current current_states
            current_states = next_states

        return self.goals[:, None].expand_as(states), states, actions, rewards, done

    @property
    def n_states(self):
        return self.grid_size**2 + 1

    def sample_goals(self, n: int):
        return torch.randint(0, self.grid_size, (n, 2))

    def step_fn(
        self,
        current_states: torch.Tensor,
        actions: torch.Tensor,
    ):
        B = self.n_tasks
        # Convert current current_states to indices
        current_state_indices = (
            current_states[:, 0] * self.grid_size + current_states[:, 1]
        )

        rewards = self.R[torch.arange(B), current_state_indices, actions]

        # Compute next state indices
        next_state_indices = torch.argmax(
            self.T[torch.arange(B), current_state_indices, actions], dim=1
        )

        # Convert next state indices to coordinates
        next_states = torch.stack(
            (
                next_state_indices // self.grid_size,
                next_state_indices % self.grid_size,
            ),
            dim=1,
        )
        return next_states, rewards, False, {}

    def visualize_policy(self, Pi, task_idx: int = 0):  # dead:disable
        self.check_pi(Pi)
        N = self.n_states
        A = len(self.deltas)

        # get the grid size
        grid_size = int(N**0.5)

        # initialize the figure
        _, ax = plt.subplots()

        # get the policy for the specified task
        pi = Pi[task_idx].numpy()

        # for each state
        for n in range(N):
            # get the policy for state n
            policy = pi[n]

            # get the x, y coordinates of the state
            x, y = n % grid_size, n // grid_size

            # for each action
            for a in range(A):
                # get the delta for action a
                dx, dy = self.deltas[a].numpy()

                # plot a line from (x, y) to (x+dx, y+dy) with color and width based on policy[a]
                color = plt.cm.viridis(policy[a].item())
                ax.plot(
                    [x, x + dx * 0.3],
                    [y, y + dy * 0.3],
                    color=color,
                    # lw=policy[a].item() * 10,
                )

        # set the axis limits and labels
        ax.set_xlim(-1, grid_size)
        ax.set_ylim(-1, grid_size)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        plt.gca().invert_yaxis()
        plt.savefig(f"policy{task_idx}.png")
