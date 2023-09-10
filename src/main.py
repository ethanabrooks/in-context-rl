import datetime
import time
import urllib

import tomli
from dollar_lambda import CommandTree, argument
from git import Repo
from ray import tune
from ray.air.integrations.wandb import setup_wandb

import wandb
from config import config
from param_space import param_space
from train import train

tree = CommandTree()


def get_project_name():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["tool"]["poetry"]["name"]


def check_dirty():
    assert not Repo(".").is_dirty()


@tree.subcommand(parsers=dict(name=argument("name")))
def log(
    name: str,
    allow_dirty: bool = False,
):
    if not allow_dirty:
        check_dirty()

    run = wandb.init(config=config, name=name, project=get_project_name())
    train(**config, run=run)


@tree.command()
def no_log():  # dead: disable
    return train(**config, run=None)


def get_git_rev():
    repo = Repo(".")
    if repo.head.is_detached:
        return repo.head.object.name_rev
    else:
        return repo.active_branch.commit.name_rev


@tree.subcommand()
def sweep(  # dead: disable
    gpus_per_proc: int,
    group: str = None,
    notes: str = None,
    num_samples: int = None,
    allow_dirty: bool = False,
):
    if group is None:
        group = datetime.datetime.now().strftime("-%d-%m-%H:%M:%S")
    commit = get_git_rev()
    project_name = get_project_name()
    if not allow_dirty:
        check_dirty()

    def trainable(sweep_params):
        sleep_time = 1
        config.update(**sweep_params)
        while True:
            try:
                run = setup_wandb(
                    config=dict(**config, commit=commit),
                    group=group,
                    project=project_name,
                    rank_zero_only=False,
                    notes=notes,
                    resume="never",
                )
                break
            except wandb.errors.CommError:
                time.sleep(sleep_time)
                sleep_time *= 2
        print(
            f"wandb: ️👪 View group at {run.get_project_url()}/groups/{urllib.parse.quote(group)}/workspace"
        )
        config.update(run=run)
        return train(**config)

    tune.Tuner(
        trainable=tune.with_resources(trainable, dict(gpu=gpus_per_proc)),
        tune_config=None if num_samples is None else dict(num_samples=num_samples),
        param_space=param_space,
    ).fit()


if __name__ == "__main__":
    tree()
