import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


def compute_policy_towards_goal(states, goals, grid_size):
    # Expand goals and states for broadcasting
    expanded_goals = goals[:, None, :]
    expanded_states = states[None, :, :]

    # Calculate the difference between each state and the goals
    diff = expanded_goals - expanded_states

    # Determine the action indices to move toward the goal for each state
    positive_x_i, positive_x_j = (diff[..., 0] > 0).nonzero(as_tuple=True)
    negative_x_i, negative_x_j = (diff[..., 0] < 0).nonzero(as_tuple=True)
    positive_y_i, positive_y_j = (diff[..., 1] > 0).nonzero(as_tuple=True)
    negative_y_i, negative_y_j = (diff[..., 1] < 0).nonzero(as_tuple=True)
    equal_i, equal_j = (diff == 0).all(-1).nonzero(as_tuple=True)

    # Initialize the actions tensor
    n_states = grid_size**2 + 1
    absorbing_state_idx = n_states - 1
    actions = torch.zeros(goals.size(0), n_states, 4)

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
    actions[:, absorbing_state_idx, 0] = 1  # Arbitrary action, since it doesn't matter
    actions[equal_i, equal_j, 0] = 1  # Arbitrary action, since it doesn't matter

    return actions


def get_trajectories(grid_size: int, n_data: int, episode_length: int, n_episodes: int):
    deltas = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])
    B = n_data
    N = grid_size**2 + 1
    A = len(deltas)
    goals = torch.randint(0, grid_size, (n_data, 2))
    states = torch.tensor([[i, j] for i in range(grid_size) for j in range(grid_size)])
    Pi = compute_policy_towards_goal(states, goals, grid_size)
    assert [*Pi.shape] == [B, N, A]

    # Compute next states for each action and state for each batch (goal)
    next_states = states[:, None] + deltas[None, :]
    next_states = torch.clamp(next_states, 0, grid_size - 1)
    S_ = next_states[..., 0] * grid_size + next_states[..., 1]  # Convert to indices

    # Determine if next_state is the goal for each batch (goal)
    is_goal = (goals[:, None] == states[None]).all(-1)

    # Modify transition to go to absorbing state if the next state is a goal
    absorbing_state_idx = N - 1
    S_ = S_[None].tile(B, 1, 1)
    S_[is_goal[..., None].expand_as(S_)] = absorbing_state_idx

    # Insert row for absorbing state
    padding = (0, 0, 0, 1)  # left 0, right 0, top 0, bottom 1
    S_ = F.pad(S_, padding, value=absorbing_state_idx)
    T = F.one_hot(S_, num_classes=N).float()
    R = is_goal.float()[..., None].tile(1, 1, A)
    R = F.pad(R, padding, value=0)  # Insert row for absorbing state

    trajectory_length = episode_length * n_episodes
    states = torch.zeros((B, trajectory_length, 2), dtype=torch.int)
    actions = torch.zeros((B, trajectory_length), dtype=torch.int)
    rewards = torch.zeros((B, trajectory_length))
    current_states = torch.randint(0, grid_size, (n_data, 2))

    for t in tqdm(range(trajectory_length), desc="Sampling trajectories"):
        if t % episode_length == 0:
            current_states = torch.randint(0, grid_size, (n_data, 2))
        # Convert current current_states to indices
        current_state_indices = current_states[:, 0] * grid_size + current_states[:, 1]

        # Sample actions from the policy
        A = (
            torch.multinomial(Pi[torch.arange(B), current_state_indices], 1)
            .squeeze(1)
            .long()
        )

        # Store the current current_states and rewards
        states[:, t] = current_states
        actions[:, t] = A
        rewards[:, t] = R[torch.arange(B), current_state_indices, A]

        # Compute next state indices
        next_state_indices = torch.argmax(
            T[torch.arange(B), current_state_indices, A], dim=1
        )

        # Convert next state indices to coordinates
        next_states = torch.stack(
            (next_state_indices // grid_size, next_state_indices % grid_size), dim=1
        )

        # Update current current_states
        current_states = next_states

    return states, actions, rewards


def round_to(tensor, decimals=2):
    return (tensor * 10**decimals).round() / (10**decimals)


def quantize_tensor(tensor, n_bins):
    # Flatten tensor
    flat_tensor = tensor.flatten()

    # Sort the flattened tensor
    sorted_tensor, _ = torch.sort(flat_tensor)

    # Determine the thresholds for each bin
    n_points_per_bin = int(math.ceil(len(sorted_tensor) / n_bins))
    thresholds = sorted_tensor[::n_points_per_bin].contiguous()

    # Assign each value in the flattened tensor to a bucket
    # The bucket number is the quantized value
    quantized_tensor = torch.bucketize(flat_tensor, thresholds)

    # Reshape the quantized tensor to the original tensor's shape
    quantized_tensor = quantized_tensor.view(tensor.shape)

    return quantized_tensor


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
        self.data = (
            torch.cat(
                [self.observations, self.actions[..., None], self.rewards[..., None]],
                dim=-1,
            )
            .long()
            .reshape(n_data, -1)
            .contiguous()
        ).cuda()
        self.mask = torch.ones_like(self.data).cuda()

    @property
    def action_dim(self):
        return 1

    @property
    def n_tokens(self):
        return 1 + self.data.max().round().long().item()

    @property
    def observation_dim(self):
        _, _, observation_dim = self.observations.shape
        return observation_dim

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]

    def __len__(self):
        return len(self.data)


def get_step_dim(observation_dim, action_dim):
    return observation_dim + action_dim + 1


def split_sequence(sequence, observation_dim, action_dim):
    transition_dim = get_step_dim(observation_dim, action_dim)
    n_batch, _ = sequence.shape
    sequence = sequence.reshape(n_batch, -1, transition_dim)

    observations = sequence[:, :, :observation_dim]
    actions = sequence[:, :, observation_dim : observation_dim + action_dim]
    rewards = sequence[:, :, -1]
    return dict(observations=observations, actions=actions, rewards=rewards)
