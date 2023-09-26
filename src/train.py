import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from matplotlib import pyplot as plt
from rich import print
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

import data
import evaluators.ad
import evaluators.adpp
import wandb
from data import Data
from models import GPT
from optimizer import configure, decay_lr
from plot import plot_accuracy
from pretty import Table
from seeding import set_seed

MODEL_FNAME = "model.tar"


def load(
    load_path: Optional[str],
    net: GPT,
    run: Optional[Run],
):
    root = run.dir if run is not None else f"/tmp/wandb{time.time()}"
    wandb.restore(MODEL_FNAME, run_path=load_path, root=root)
    state = torch.load(os.path.join(root, MODEL_FNAME))
    net.load_state_dict(state, strict=True)


def evaluate(
    dataset: Data, evaluator: evaluators.ad.Evaluator, net: GPT, section: str, **kwargs
):
    df = pd.DataFrame.from_records(
        list(evaluator.evaluate(dataset=dataset, net=net, **kwargs))
    )

    metrics = df.drop(["history"], axis=1).groupby("episode").mean().metric
    graph = dataset.render_eval_metrics(*metrics)
    print("\n" + "\n".join(graph), end="\n\n")
    fig = dataset.plot_eval_metrics(df=df)
    *_, final_metric = metrics
    metric_log = {f"{section}/final {dataset.eval_metric_name}": final_metric}
    fig_log = {f"{section}/{dataset.eval_metric_name}": wandb.Image(fig)}
    return metric_log, fig_log


def train(
    adpp_args: dict,
    data_args: dict,
    data_path: Path,
    evaluate_args: dict,
    evaluate_ad_args: dict,
    evaluate_adpp_args: dict,
    grad_norm_clip: float,
    load_path: Optional[str],
    log_interval: int,
    log_tables_interval: int,
    lr: float,
    metrics_args: dict,
    model_args: dict,
    n_batch: int,
    n_epochs: int,
    optimizer_config: dict,
    run: Optional[Run],
    save_interval: int,
    seed: int,
    test_ad_interval: int,
    test_adpp_interval: int,
    weights_args: dict,
) -> None:
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

    optimizer = configure(lr=lr, module=net, **optimizer_config)

    counter = Counter()
    n_tokens = 0
    save_count = 0
    ad_log = {}
    adpp_log = {}
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
            if load_path is None:
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
            if t % log_interval == 0:
                log = {k: v / log_interval for k, v in counter.items()}
                log.update(
                    epoch=e,
                    lr=decayed_lr,
                    save_count=save_count,
                    time=(time.time() - tick) / log_interval,
                )
                counter = Counter()
                tick = time.time()
                row = dict(step=step, **log)
                log = {f"train/{k}": v for k, v in log.items()}

                # test
                log_t = t // log_interval
                if (
                    test_adpp_interval is not None
                    and (1 + log_t) % test_adpp_interval == 0
                ):
                    adpp_log, fig = evaluate(
                        dataset=dataset,
                        evaluator=evaluators.adpp.Evaluator(**adpp_args),
                        net=net,
                        section="eval AD++",
                        **evaluate_args,
                        **evaluate_adpp_args,
                    )
                    log.update(fig)
                    log_table.print_header(row)
                if log_t % test_ad_interval == 0:
                    ad_log, fig = evaluate(
                        dataset=dataset,
                        evaluator=evaluators.ad.Evaluator(),
                        net=net,
                        section="eval AD",
                        **evaluate_args,
                        **evaluate_ad_args,
                    )
                    log.update(fig)
                    log_table.print_header(row)

                log.update(adpp_log)
                log.update(ad_log)

                if log_t % log_tables_interval == 0:

                    def get_figures():
                        for name, xs in tables.items():
                            fig = plot_accuracy(*xs, name=name, ymin=0, ymax=1)
                            yield f"train/{name}", wandb.Image(fig)

                    log.update(dict(get_figures()))

                if run is not None:
                    wandb.log(log, step=step)
                plt.close()
                log_table.print_row(row)

            # save
            if t % save_interval == 0:
                save(run, net)
                save_count += 1

    save(run, net)


def save(run: Run, net: GPT):
    if run is not None:
        savepath = os.path.join(run.dir, MODEL_FNAME)
        torch.save(net.state_dict(), savepath)
        wandb.save(savepath)
