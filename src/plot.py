import matplotlib.pyplot as plt
import pandas as pd


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
    x = means.index + 1  # 1-indexed
    ax.fill_between(x, means - sems, means + sems, alpha=0.2)
    ax.plot(x, means)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("episode")
    ax.set_ylabel(name)
    ax.grid(True)
    return fig
