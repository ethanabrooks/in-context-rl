import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gym.spaces import Discrete, MultiDiscrete, Space
from tqdm import tqdm

from data.base import Data
from envs.parallel.dummy_vec_env import DummyVecEnv
from envs.parallel.subproc_vec_env import SubprocVecEnv
from models import GPT


def get_return(*rewards: float, gamma: float) -> float:
    actual_return = 0
    for r in rewards[::-1]:
        actual_return = r + gamma * actual_return
    return actual_return


def get_dim(space: Space) -> int:
    if isinstance(space, Discrete):
        return 1
    elif isinstance(space, MultiDiscrete):
        return space.nvec.size
    else:
        raise NotImplementedError


def get_metric(
    info,
    rewards,
    gamma,
):
    actual_return = get_return(*rewards, gamma=gamma)
    optimal = info.get("optimal", None)
    if optimal is None:
        return "return", actual_return
    else:
        optimal_return = get_return(*optimal, gamma=gamma)
        regret = optimal_return - actual_return
        return "regret", regret


def evaluate(dataset: Data, dummy_vec_env: bool, n_rollouts: int, **kwargs):
    N = n_rollouts
    env_fns = [dataset.build_env for _ in range(N)]
    envs: SubprocVecEnv
    envs = DummyVecEnv(env_fns) if dummy_vec_env else SubprocVecEnv(env_fns)
    try:
        yield from rollout(dataset=dataset, envs=envs, n_rollouts=n_rollouts, **kwargs)
    finally:
        envs.close()


def rollout(
    dataset: Data, envs: SubprocVecEnv, gamma: float, net: GPT, n_rollouts: int
):
    N = n_rollouts

    # task
    task = torch.tensor(envs.get_task()).cuda()
    assert [*task.shape] == [N, 2]
    if not dataset.include_goal:
        task = torch.zeros_like(task)
    task_dim = get_dim(envs.task_space)
    assert [*task.shape] == [N, task_dim]

    # observation
    observation = envs.reset()
    O = get_dim(envs.observation_space)
    assert [*observation.shape] == [N, O]

    # actions
    action_dim = get_dim(envs.action_space)
    A = envs.action_space.n
    dummy_action = torch.tensor(dataset.pad_value).repeat(N, 1).cuda()

    # reward
    dummy_reward = torch.tensor(dataset.pad_value).repeat(N, 1).cuda()

    T = dataset.episode_length * dataset.episodes_per_rollout
    tasks = torch.zeros(N, T, task_dim).cuda()
    observations = torch.zeros(N, T, O).cuda()
    actions = torch.zeros(N, T).cuda()
    rewards = torch.zeros(N, T).cuda()
    episode_count = np.zeros(N, dtype=int)
    episode_rewards = np.zeros((N, dataset.episode_length))
    episode_t = np.zeros(N, dtype=int)

    for t in tqdm(range(T)):
        tasks[:, t] = task
        observations[:, t] = torch.tensor(observation).cuda()

        # create sequence
        sequence = [
            tasks[:, : t + 1],
            observations[:, : t + 1],
            torch.cat([actions[:, :t], dummy_action], dim=1),
            torch.cat([rewards[:, :t], dummy_reward], dim=1),
        ]

        ## create context and pad
        ctx = dataset.cat_sequence(*sequence)
        assert [*ctx.shape] == [N, (t + 1) * dataset.step_dim]
        pad_size = 1 + net.context_size - ctx.numel() // N
        ctx = F.pad(ctx, (pad_size, 0), value=dataset.pad_value)

        logits: torch.Tensor
        logits, _ = net.forward(ctx)
        assert [*logits.shape] == [N, net.context_size, 1 + dataset.n_tokens]

        # zero out invalid actions
        logits = logits[:, :-1]  # exclude reward prediction
        logits = logits[:, -action_dim:]  # only consider last action
        valid = logits[:, :, :A]
        invalid = -1e8 * torch.ones_like(logits[:, :, A:])

        # sample action
        logits = torch.cat([valid, invalid], dim=-1)
        probs = logits.softmax(dim=-1)
        assert [*probs.shape] == [N, 1, dataset.n_tokens + 1]
        [action] = torch.multinomial(probs.squeeze(1), num_samples=1).T

        observation, reward, done, info = envs.step(action.cpu().numpy())
        assert [*observation.shape] == [N, O]
        assert [*reward.shape] == [N]
        assert [*done.shape] == [N]
        assert len(info) == N
        actions[:, t] = action
        rewards[:, t] = torch.tensor(reward).cuda()
        episode_rewards[np.arange(N), episode_t] = reward
        episode_t += 1

        for n, (d, ec, er, et, i) in enumerate(
            zip(done, episode_count, episode_rewards, episode_t, info)
        ):
            assert isinstance(d, (bool, np.bool_))
            assert isinstance(i, dict)
            if d:
                name, x = get_metric(info=i, rewards=er[:et], gamma=gamma)
                yield dict(n=n, name=name, t=ec, metric=x)
                episode_rewards[n] = 0
                episode_count[n] += 1
                episode_t[n] = 0
                observation[n] = envs.reset(n)


def plot_returns(df: pd.DataFrame, name: str, ymin: float, ymax: float):
    means = df.groupby("t").metric.mean()
    sems = df.groupby("t").metric.sem()

    fig, ax = plt.subplots()
    ax.fill_between(means.index, means - sems, means + sems, alpha=0.2)
    ax.plot(means.index, means)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("episode")
    ax.set_ylabel(name)
    ax.grid(True)
    return fig
