# In-Context RL with Model-Based Planning
This is ongoing research that I am conducting as part of my PhD at University of Michigan.

## Abstract

In this paper, we propose an extension to [Algorithm Distillation
(AD)](https://arxiv.org/abs/2210.14215) which augments the AD model with
world-model predictions of states and rewards. We use these predictions to
simulate rollouts during evaluation, which we use to estimate values. We then
use these estimates to greedily choose actions. We demonstrate that this
approach generally outperforms naive Algorithm Distillation and is far more
capable of generalization to novel tasks. We also provide some analysis of the
model's capability to scale with the quantity of data in the dataset.

## Overleaf (in-progress)
https://www.overleaf.com/read/hfdgqmrkzqtk

## Installation
### Poetry
- Install [poetry](https://python-poetry.org/docs/#installation)
- Run `poetry install` to install dependencies
- Run `poetry shell` to activate the virtual environment

### Nix
- Install [nix](https://nixos.org/download.html)
- Activate [flakes](https://nixos.wiki/wiki/Flakes)
- Run `nix develop` to activate the virtual environment.
- Alternatively, if you have [`direnv`](https://direnv.net/) installed, you can run `direnv allow` to activate the virtual environment.


## Experiments
5x5 Gridworld with AD and AD++:
```
python src/main.py --config grid_world/adpp5x5
```

5x5 Gridworld (dense reward) with AD and AD++:
```
python src/main.py --config grid_world/adpp5x5dense
```

To activate logging via [`wandb`](https://wandb.ai/site/), use
`python src/main.py log "Name of experiment" --config ...`
