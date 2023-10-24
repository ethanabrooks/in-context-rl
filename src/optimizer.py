import math


def decay_lr(lr: float, final: int, step: int, warmup: int):
    if step < warmup:
        # linear warmup
        lr_mult = float(step) / float(max(1, warmup))
    else:
        # cosine learning rate decay
        progress = float(step - warmup) / float(max(1, final - warmup))
        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr * lr_mult
