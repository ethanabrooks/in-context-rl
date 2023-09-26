from pathlib import Path
from typing import Optional

from dollar_lambda import CommandTree, argument
from rich import print
from wandb.sdk.wandb_run import Run

import data
import evaluators.ad
import evaluators.adpp
import wandb
from main import check_dirty, get_project_name
from models import GPT
from seeding import set_seed
from train import evaluate, load

tree = CommandTree()


def main(
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
    net = GPT(
        encoder=dataset.encoder,
        n_tokens=dataset.n_tokens,
        step_dim=dataset.step_dim,
        **model_args,
    )
    if load_path is not None:
        load(load_path, net, run)
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
    main(**loaded_config, run=run)


@tree.command(parsers=dict(load_path=argument("load_path")))
def no_log(
    load_path: str,
    algo: str = "AD++",
    dummy_vec_env: bool = False,
):
    config: dict = wandb.Api().run(load_path).config
    config.update(algo=algo, load_path=load_path)
    evaluator_args = config["evaluate_args"]
    assert isinstance(evaluator_args, dict)
    evaluator_args.update(dummy_vec_env=dummy_vec_env)
    return main(**config, run=None)


if __name__ == "__main__":
    tree()
