from collections import Counter
import os
import random
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from wandb.sdk.wandb_run import Run

import wandb
from data import RLData, unwrap
from models import GPT
from pretty import print_row


def evaluate(net: nn.Module, test_loader: DataLoader, **kwargs):
    net.eval()
    counter = Counter()
    dataset = unwrap(test_loader.dataset)
    with torch.no_grad():
        for sequence, mask in test_loader:
            logits, loss = net(sequence, mask)
            log = dataset.get_metrics(sequence=sequence, logits=logits, **kwargs)
            counter.update(dict(**log, loss=loss.item()))
    log = {k: (v / len(test_loader)) for k, v in counter.items()}
    return log


def train(
    data_args: dict,
    log_freq: int,
    lr: float,
    metrics_args: dict,
    model_args: dict,
    n_batch: int,
    n_epochs: int,
    n_steps: int,
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

    dataset = RLData(**data_args, n_data=n_steps)

    print("Create net... ", end="", flush=True)
    net = GPT(n_tokens=dataset.n_tokens, step_dim=dataset.step_dim, **model_args).cuda()
    print("✓")

    # Split the dataset into train and test sets
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    counter = Counter()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    for e in range(n_epochs):
        # Split the dataset into train and test sets
        train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=n_batch, shuffle=False)
        for t, (sequence, mask) in enumerate(train_loader):
            step = e * len(train_loader) + t
            if t % test_freq == 0:
                log = evaluate(net=net, test_loader=test_loader, **metrics_args)
                print_row(log, show_header=True)
                if run is not None:
                    wandb.log({f"test/{k}": v for k, v in log.items()}, step=step)

            if t == int(0.5 * n_steps):
                optimizer.param_groups[0]["lr"] *= 0.1
            net.train()
            optimizer.zero_grad()
            weights = dataset.weights(sequence.shape, **weights_args)
            logits, loss = net(sequence, mask, weights)
            log = dataset.get_metrics(sequence=sequence, logits=logits, **metrics_args)
            counter.update(dict(**log, loss=loss.item()))

            loss.backward()
            optimizer.step()

            if t % log_freq == 0:
                log = {k: v / log_freq for k, v in counter.items()}
                counter = Counter()
                print_row(log, show_header=(t % test_freq == 0))
                if run is not None:
                    wandb.log({f"train/{k}": v for k, v in log.items()}, step=step)

            if t % save_freq == 0:
                torch.save(
                    {"state_dict": net.state_dict()},
                    os.path.join(save_dir, "model.tar"),
                )

    torch.save({"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar"))
