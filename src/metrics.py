import torch

from data import split_sequence


def get_metrics(sequence, logits, observation_dim, action_dim):
    n_batch, seq_len = sequence.shape
    n_batch2, seq_len2, _ = logits.shape
    assert n_batch == n_batch2
    assert seq_len == seq_len2 + 1

    prefix = sequence[:, :1]
    preds = torch.cat([prefix, logits.argmax(-1)], dim=1)
    tgts = sequence
    acc = {}
    final_acc = {}
    for (name, pred), (name2, tgt) in zip(
        dict(**split_sequence(preds, observation_dim, action_dim), total=preds).items(),
        dict(**split_sequence(tgts, observation_dim, action_dim), total=tgts).items(),
    ):
        assert name == name2
        acc[name] = pred == tgt
        if pred.ndim == 2:
            pred = pred[..., None]
        pred = get_final(pred, n_batch)
        tgt = get_final(tgt, n_batch)
        final_acc[name] = pred == tgt
    logs = dict(
        **{f"{k} accuracy": v for k, v in acc.items()},
        **{f"final {k} accuracy": v for k, v in final_acc.items()},
    )
    return {k: v.float().mean().item() for k, v in logs.items()}


def get_final(x, n):
    if x.ndim == 2:
        x = x[..., None]
    x = x.reshape(n, -1, x.size(-1))
    _, n, *_ = x.shape
    return x[:, -n:]
