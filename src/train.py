import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from rich import print
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

import data
import wandb
from evaluators.ad import Evaluator
from models import GPT
from optimizer import configure, decay_lr
from plot import plot_accuracy, plot_returns
from pretty import Table, render_graph
from utils import set_seed


def train(
    data_args: dict,
    data_path: Path,
    evaluate_args: dict,
    grad_norm_clip: float,
    log_freq: int,
    log_tables_freq: int,
    lr: float,
    metrics_args: dict,
    model_args: dict,
    n_batch: int,
    n_epochs: int,
    optimizer_config: dict,
    run: Optional[Run],
    run_name: str,
    save_freq: int,
    seed: int,
    test_freq: int,
    weights_args: dict,
) -> None:
    save_dir = os.path.join("results", run_name)
    set_seed(seed)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    dataset = data.make(data_path, **data_args)

    print("Create net... ", end="", flush=True)
    net = GPT(n_tokens=dataset.n_tokens, step_dim=dataset.step_dim, **model_args).cuda()
    print("✓")

    optimizer = configure(lr=lr, module=net, **optimizer_config)

    counter = Counter()
    n_tokens = 0
    eval_log = None
    tick = time.time()
    log_table = Table()

    for e in range(n_epochs):
        # Split the dataset into train and test sets
        loader = DataLoader(dataset, batch_size=n_batch, shuffle=True)
        print("Loading train data... ", end="", flush=True)
        for t, (sequence, mask) in enumerate(loader):
            sequence = sequence.cuda()
            mask = mask.cuda()
            step = e * len(loader) + t

            # gradient update
            net.train()
            optimizer.zero_grad()
            weights = dataset.weights(sequence.shape, **weights_args)
            logits, loss = net.forward(sequence, mask, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_norm_clip)
            optimizer.step()

            # update learning rate
            n_tokens += mask.sum()
            final_tokens = (
                n_epochs * len(loader) * n_batch * dataset.step_dim
            )  # number of tokens seen during training
            decayed_lr = decay_lr(lr, final_tokens=final_tokens, n_tokens=n_tokens)
            for param_group in optimizer.param_groups:
                param_group.update(lr=decayed_lr)

            # log
            log, tables = dataset.get_metrics(
                logits=logits, mask=mask, sequence=sequence, **metrics_args
            )
            counter.update(dict(**log, loss=loss.item()))
            if t % log_freq == 0:
                log = {k: v / log_freq for k, v in counter.items()}
                log.update(epoch=e, lr=decayed_lr, time=(time.time() - tick) / log_freq)
                counter = Counter()
                tick = time.time()
                row = dict(step=step, **log)
                log = {f"train/{k}": v for k, v in log.items()}

                # test
                log_t = t // log_freq
                if log_t % test_freq == 0:
                    df = pd.DataFrame.from_records(
                        list(
                            Evaluator.evaluate(
                                dataset=dataset, net=net, **evaluate_args
                            )
                        )
                    )

                    min_return, max_return = dataset.return_range
                    metrics = df.drop("name", axis=1).groupby("t").mean().metric
                    graph = render_graph(*metrics, max_num=max_return)
                    print("\n" + "\n".join(graph), end="\n\n")
                    try:
                        [name] = df.name.unique()
                    except ValueError:
                        raise ValueError("Multiple names in the same rollout")
                    fig = plot_returns(
                        df=df,
                        name=name,
                        ymin=min_return,
                        ymax=max_return,
                    )
                    *_, final_metric = metrics
                    log.update({f"eval/{name}": wandb.Image(fig)})
                    eval_log = {f"eval/final {name}": final_metric}
                    log_table.print_header(row)

                if eval_log is not None:
                    log.update(eval_log)

                if log_t % log_tables_freq == 0:

                    def get_figures():
                        for name, xs in tables.items():
                            fig = plot_accuracy(*xs, name=name, ymin=0, ymax=1)
                            yield f"train/{name}", wandb.Image(fig)

                    log.update(dict(get_figures()))

                if run is not None:
                    wandb.log(log, step=step)
                log_table.print_row(row)

            # save
            if t % save_freq == 0:
                torch.save(
                    {"state_dict": net.state_dict()},
                    os.path.join(save_dir, "model.tar"),
                )

    torch.save({"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar"))
