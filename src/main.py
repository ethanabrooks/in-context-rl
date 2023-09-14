import datetime
import time
import urllib
from pathlib import Path

import tomli
from dollar_lambda import CommandTree, argument, option
from git import Repo
from omegaconf import DictConfig, OmegaConf
from ray import tune
from ray.air.integrations.wandb import setup_wandb
from rich import print

import wandb
from param_space import param_space
from train import train

tree = CommandTree()


def is_alphabetical_order(d, parent_keys=None):
    """Recursively check if all keys in the dictionary are in alphabetical order
    and return the keys that are out of order.
    """
    if parent_keys is None:
        parent_keys = []
    if not isinstance(d, dict):
        return []
    keys = list(d.keys())
    out_of_order_keys = [
        parent_keys + [k1] for k1, k2 in zip(keys, sorted(keys)) if k1 != k2
    ]
    for k, v in d.items():
        out_of_order_keys.extend(is_alphabetical_order(v, parent_keys + [k]))
    return out_of_order_keys


def check_alphabetical_order(d: DictConfig, name: str):
    out_of_order_keys = is_alphabetical_order(OmegaConf.to_container(d))
    if out_of_order_keys:
        print(f"The following keys are not in alphabetical order in {name}:")
        for keys in out_of_order_keys:
            print(".".join(keys))
        exit(1)


def get_config(config_name):
    root = Path("configs")
    config_path = root / f"{config_name}.yml"
    config = OmegaConf.load(config_path)
    check_alphabetical_order(config, str(config_path))
    base_config_path = root / "base.yml"
    base_config = OmegaConf.load(base_config_path)
    check_alphabetical_order(base_config, str(base_config_path))
    merged = OmegaConf.merge(base_config, config)
    resolved = OmegaConf.to_container(merged, resolve=True)
    return resolved


def get_data_path(name):
    path = Path("src", "data") / name
    path = path.with_suffix(".py")
    assert path.exists()
    return path


def get_project_name():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["tool"]["poetry"]["name"]


def check_dirty():
    assert not Repo(".").is_dirty()


parsers = dict(config=option("config", default="full_history"))


@tree.subcommand(parsers=dict(name=argument("name"), **parsers))
def log(
    name: str,
    config: str,
    allow_dirty: bool = False,
):
    if not allow_dirty:
        check_dirty()

    config_name = config
    data_path = get_data_path(config)
    config = get_config(config)
    run = wandb.init(
        config=dict(**config, config=config_name), name=name, project=get_project_name()
    )
    train(**config, data_path=data_path, run=run)


@tree.command(parsers=parsers)
def no_log(config):  # dead: disable
    return train(**get_config(config), data_path=get_data_path(config), run=None)


def get_git_rev():
    repo = Repo(".")
    if repo.head.is_detached:
        return repo.head.object.name_rev
    else:
        return repo.active_branch.commit.name_rev


@tree.subcommand(parsers=parsers)
def sweep(  # dead: disable
    config: str,
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
    data_path = get_data_path(config)
    config_name = config
    config = get_config(config)
    if not allow_dirty:
        check_dirty()

    def trainable(sweep_params):
        sleep_time = 1
        config.update(**sweep_params)
        while True:
            try:
                run = setup_wandb(
                    config=dict(**config, commit=commit, config=config_name),
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
        config.update(run=run, data_path=data_path)
        return train(**config)

    tune.Tuner(
        trainable=tune.with_resources(trainable, dict(gpu=gpus_per_proc)),
        tune_config=None if num_samples is None else dict(num_samples=num_samples),
        param_space=param_space,
    ).fit()


if __name__ == "__main__":
    tree()
