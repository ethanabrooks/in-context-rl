from itertools import islice

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
        dict(i=i, t=t, returns=r)
        for i, rollout in enumerate(rollouts)
        for t, r in enumerate(rollout)
    ]
    return pd.DataFrame(records)


def rollout(
    dataset: Data,
    gamma: float,
    net: GPT,
):
    env = dataset.build_env()
    task = env.get_task().cuda()
    if not dataset.include_goal:
        task = torch.zeros_like(task)
    observation = env.reset().cuda()

    action_dim = 1
    act_card = env.action_space.n

    history = []
    total_reward = 0

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

        # zero out invalid actions
        logits = logits[:-1]  # exclude reward prediction
        logits = logits[-action_dim:]  # only consider last action
        valid = logits[:, :act_card]
        invalid = -1e8 * torch.ones_like(logits[:, act_card:])

        # sample action
        logits = torch.cat([valid, invalid], dim=-1)
        probs = logits.softmax(dim=-1)
        [action] = torch.multinomial(probs, num_samples=1).T

        observation, reward, done, _ = env.step(action.cpu())
        total_reward = gamma * total_reward + reward
        if done:
            yield total_reward.item()
            total_reward = 0
            observation = env.reset().cuda()

        # extend history
        observation = observation.cuda()
        reward = reward[None].long().cuda()  # TODO! remove long
        history.extend([action, reward])


def plot(ymin: float, ymax: float, df: pd.DataFrame):
    means = df.groupby("t").returns.mean()
    sems = df.groupby("t").returns.sem()

    fig, ax = plt.subplots()
    ax.fill_between(means.index, means - sems, means + sems, alpha=0.2)
    ax.plot(means.index, means)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("episode")
    ax.set_ylabel("returns")
    ax.grid(True)
    return fig
