import os
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from rich import print
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

import data
import wandb
from data.base import unwrap
from models import GPT
from optimizer import configure, decay_lr
from pretty import print_row


def evaluate(net: nn.Module, test_loader: DataLoader, **kwargs):
    net.eval()
    counter = Counter()
    dataset = unwrap(test_loader.dataset)
    with torch.no_grad():
        for sequence, mask in tqdm(test_loader, desc="Evaluating"):
            logits, loss = net(sequence, mask)
            log = dataset.get_metrics(
                logits=logits, mask=mask, sequence=sequence, **kwargs
            )
            counter.update(dict(**log, loss=loss.item()))
    log = {k: (v / len(test_loader)) for k, v in counter.items()}
    return log


def train(
    data_args: dict,
    data_path: Path,
    grad_norm_clip: float,
    log_freq: int,
    lr: float,
    metrics_args: dict,
    model_args: dict,
    n_batch: int,
    n_epochs: int,
    n_steps: int,
    optimizer_config: dict,
    run: Optional[Run],
    run_name: str,
    save_freq: int,
    seed: int,
    test_split: float,
    test_freq: int,
    weights_args: dict,
) -> None:
    save_dir = os.path.join("results", run_name)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # If you are using CUDA (GPU), you also need to set the seed for the CUDA device
    # This ensures reproducibility for GPU calculations as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for Python's random module
    random.seed(seed)

    dataset = data.make(data_path, **data_args, n_data=n_steps)

    print("Create net... ", end="", flush=True)
    net = GPT(n_tokens=dataset.n_tokens, step_dim=dataset.step_dim, **model_args).cuda()
    print("✓")

    optimizer = configure(lr=lr, module=net, **optimizer_config)

    # Split the dataset into train and test sets
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    print("Splitting data... ", end="", flush=True)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print("✓")

    counter = Counter()
    n_tokens = 0

    for e in range(n_epochs):
        # Split the dataset into train and test sets
        train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=n_batch, shuffle=False)
        print("Loading train data... ", end="", flush=True)
        for t, (sequence, mask) in enumerate(train_loader):
            step = e * len(train_loader) + t

            # test
            if t % test_freq == 0:
                log = evaluate(net=net, test_loader=test_loader, **metrics_args)
                print_row(log, show_header=True)
                if run is not None:
                    wandb.log({f"test/{k}": v for k, v in log.items()}, step=step)

            # gradient update
            net.train()
            optimizer.zero_grad()
            weights = dataset.weights(sequence.shape, **weights_args)
            logits, loss = net(sequence, mask, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_norm_clip)
            optimizer.step()

            # update learning rate
            n_tokens += mask.sum()
            final_tokens = (
                n_epochs * len(train_loader) * n_batch * dataset.step_dim
            )  # number of tokens seen during training
            decayed_lr = decay_lr(lr, final_tokens=final_tokens, n_tokens=n_tokens)
            for param_group in optimizer.param_groups:
                param_group.update(lr=decayed_lr)

            # log
            log = dataset.get_metrics(
                logits=logits, mask=mask, sequence=sequence, **metrics_args
            )
            counter.update(dict(**log, loss=loss.item()))
            if t % log_freq == 0:
                log = {k: v / log_freq for k, v in counter.items()}
                log.update(lr=decayed_lr)
                counter = Counter()
                print_row(log, show_header=(t % test_freq == 0))
                if run is not None:
                    wandb.log({f"train/{k}": v for k, v in log.items()}, step=step)

            # save
            if t % save_freq == 0:
                torch.save(
                    {"state_dict": net.state_dict()},
                    os.path.join(save_dir, "model.tar"),
                )

    torch.save({"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar"))
