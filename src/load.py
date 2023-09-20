import os
from pathlib import Path
from typing import Optional

import torch
from dollar_lambda import CommandTree, argument
from rich import print
from wandb.sdk.wandb_run import Run

import data
import evaluators.ad
import evaluators.adpp
import wandb
from main import check_dirty, get_project_name
from models import GPT
from train import MODEL_FNAME, evaluate
from utils import set_seed

tree = CommandTree()


def load(
    adpp_args: dict,
    algo: str,
    data_args: dict,
    data_path: Path,
    evaluate_args: dict,
    load_path: Optional[str],
    model_args: dict,
    run: Optional[Run],
    seed: int,
    **_,
):
    set_seed(seed)

    dataset = data.make(data_path, **data_args)
    print("Create net... ", end="", flush=True)
    net = GPT(n_tokens=dataset.n_tokens, step_dim=dataset.step_dim, **model_args)
    if load_path is not None:
        root = run.dir if run is not None else "/tmp"
        wandb.restore(MODEL_FNAME, run_path=load_path, root=root)
        state = torch.load(os.path.join(root, MODEL_FNAME))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        net.load_state_dict(state["state_dict"], strict=True)
    net = net.cuda()
    print("✓")

    if algo == "AD":
        evaluator = evaluators.ad.Evaluator()
        section = "eval AD"
    elif algo == "AD++":
        evaluator = evaluators.adpp.Evaluator(**adpp_args)
        section = "eval AD++"
    else:
        raise ValueError(f"Invalid algorithm: {algo}")

    _, fig_log = evaluate(
        dataset=dataset, evaluator=evaluator, net=net, section=section, **evaluate_args
    )
    if run is not None:
        wandb.log(fig_log)


@tree.subcommand(parsers=dict(name=argument("name"), load_path=argument("load_path")))
def log(
    load_path: str,
    name: str,
    algo: str = "AD++",
    allow_dirty: bool = False,
):
    if not allow_dirty:
        check_dirty()

    config = dict(algo=algo, load_path=load_path)
    run = wandb.init(
        config=config,
        name=name,
        project=get_project_name(),
    )
    loaded_config = wandb.Api().run(load_path).config
    loaded_config.update(config)
    load(**loaded_config, run=run)


@tree.command(parsers=dict(load_path=argument("load_path")))
def no_log(
    load_path: str,
    algo: str = "AD++",
):
    loaded_config = wandb.Api().run(load_path).config
    loaded_config.update(algo=algo, load_path=load_path)
    return load(**loaded_config, run=None)


if __name__ == "__main__":
    tree()
