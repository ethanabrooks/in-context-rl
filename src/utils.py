import math


def decay_lr(lr: float, final_tokens: int, n_tokens: int):
    warmup_tokens = final_tokens // 20
    if n_tokens < warmup_tokens:
        # linear warmup
        lr_mult = float(n_tokens) / float(max(1, warmup_tokens))
    else:
        breakpoint()
        # cosine learning rate decay
        progress = float(n_tokens - warmup_tokens) / float(
            max(1, final_tokens - warmup_tokens)
        )
        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr * lr_mult
