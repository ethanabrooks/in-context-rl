import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def plot_accuracy(
    *accuracies: float,
    name: str,
    ymin: float,
    ymax: float,
):
    fig, ax = plt.subplots()
    x = list(range(len(accuracies)))
    ax.bar(x, accuracies)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("context step")
    ax.set_ylabel(f"{name} accuracy")
    ax.grid(True)
    return fig


def plot_eval_metrics(df: pd.DataFrame, name: str, ymin: float, ymax: float):
    means = df.groupby("episode").metric.mean()
    sems = df.groupby("episode").metric.sem()

    fig, ax = plt.subplots()
    ax.fill_between(means.index, means - sems, means + sems, alpha=0.2)
    ax.plot(means.index, means)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("episode")
    ax.set_ylabel(name)
    ax.grid(True)
    return fig


def plot_rollout(rollout: np.ndarray):
    # Extract goals, states, actions, and rewards from the sequence
    states = rollout.reshape(-1, 2)[1::3]
    actions = rollout[4::6]
    rewards = rollout[5::6]
    deltas = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    x, y = states.T

    # Using indices as the parameter
    t = np.arange(len(x))

    # Create splines for x(t) and y(t)
    sx = CubicSpline(t, x)
    sy = CubicSpline(t, y)

    # Generate new t values
    tnew = np.linspace(0, len(x) - 1, 300)

    _, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sx(tnew), sy(tnew), "r-", label="Smoothed curve")
    ax.plot(x, y, "bo", label="Original points")

    for i, (xi, yi) in enumerate(zip(x, y)):
        xi += np.random.rand() / 10
        yi += np.random.rand() / 10
        ax.text(xi, yi, str(i), fontsize=12)

    # Create a colormap and normalize rewards for colormap
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=min(rewards), vmax=max(rewards))

    # Draw arrows based on the deltas
    for action, reward, xi, yi in itertools.zip_longest(actions, rewards, x, y):
        if action is not None:
            if reward is None:
                color = "black"
            else:
                color = cmap(norm(reward))
            ax.arrow(
                xi,
                yi,
                *(0.1 * deltas[action]),
                head_width=0.1,
                head_length=0.2,
                fc=color,
                ec=color,
            )

    plt.grid(True)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Reward")
    plt.savefig("rollout.png")


plot_rollout  # whitelist
