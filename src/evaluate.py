from itertools import islice
import gym

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.base import Data
from models import GPT


def evaluate(dataset: Data, net: GPT, n_rollouts: int, **kwargs):
    rollouts = [
        list(
            islice(
                rollout(dataset=dataset, net=net, **kwargs),
                dataset.episodes_per_rollout,
            )
        )
        for _ in tqdm(range(n_rollouts))
    ]
    records = [
        dict(i=i, t=t, name=n, returns=r)
        for i, rollout in enumerate(rollouts)
        for t, (n, r) in enumerate(rollout)
    ]
    return pd.DataFrame(records)


def get_return(*rewards: float, gamma: float):
    actual_return = 0.0
    for r in rewards[::-1]:
        actual_return = r + gamma * actual_return
    return actual_return


def get_dim(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Discrete):
        return 1
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return space.nvec.size
    else:
        raise NotImplementedError


def rollout(
    dataset: Data,
    gamma: float,
    net: GPT,
):
    env = dataset.build_env()

    # task
    task = env.get_task().cuda()
    assert [*task.shape] == [2]
    if not dataset.include_goal:
        task = torch.zeros_like(task)
    task_dim = get_dim(env.task_space)
    assert [*task.shape] == [task_dim]

    # observation
    observation = env.reset().cuda()
    observation_dim = get_dim(env.observation_space)
    assert [*observation.shape] == [observation_dim]

    action_dim = get_dim(env.action_space)
    act_card = env.action_space.n

    history = []
    rewards = []

    while True:
        history.extend([task, observation])
        ## create context and pad
        ctx = torch.cat(history)
        pad_value = dataset.pad_value
        ctx = F.pad(ctx, (0, 1 + action_dim), value=pad_value)  # add dummy action
        pad_size = 1 + net.context_size - ctx.numel()
        ctx = F.pad(ctx, (pad_size, 0), value=pad_value)
        ctx = ctx[None].cuda()

        [logits], _ = net.forward(ctx)
        assert [*logits.shape] == [net.context_size, dataset.n_tokens + 1]

        # zero out invalid actions
        logits = logits[:-1]  # exclude reward prediction
        logits = logits[-action_dim:]  # only consider last action
        valid = logits[:, :act_card]
        invalid = -1e8 * torch.ones_like(logits[:, act_card:])

        # sample action
        logits = torch.cat([valid, invalid], dim=-1)
        probs = logits.softmax(dim=-1)
        assert [*probs.shape] == [1, dataset.n_tokens + 1]
        [action] = torch.multinomial(probs, num_samples=1).T

        observation, reward, done, info = env.step(action.cpu())
        assert [*observation.shape] == [observation_dim]
        assert [*reward.shape] == []
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        rewards.append(reward)
        if done:
            actual_return = get_return(*rewards, gamma=gamma)
            optimal = info.get("optimal", None)
            if optimal is None:
                yield "return", actual_return.item()
            else:
                optimal_return = get_return(*optimal, gamma=gamma)
                regret = optimal_return - actual_return
                yield "regret", regret.item()
            rewards = []
            observation = env.reset().cuda()

        # extend history
        observation = observation.cuda()
        reward = reward[None].long().cuda()  # TODO! remove long
        history.extend([action, reward])


def plot_returns(df: pd.DataFrame, name: str, ymin: float, ymax: float):
    means = df.groupby("t").returns.mean()
    sems = df.groupby("t").returns.sem()

    fig, ax = plt.subplots()
    ax.fill_between(means.index, means - sems, means + sems, alpha=0.2)
    ax.plot(means.index, means)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("episode")
    ax.set_ylabel(name)
    ax.grid(True)
    return fig
