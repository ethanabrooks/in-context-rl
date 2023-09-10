import torch

from data import split_sequence


def get_metrics(
    action_dim: int,
    logits: torch.Tensor,
    graphs_per_component: int,
    observation_dim: int,
    sequence: torch.Tensor,
):
    n_batch, seq_len = sequence.shape
    n_batch2, seq_len2, _ = logits.shape
    assert n_batch == n_batch2
    assert seq_len == seq_len2 + 1

    prefix = sequence[:, :1]
    preds = torch.cat([prefix, logits.argmax(-1)], dim=1)
    tgts = sequence
    split_preds = split_sequence(preds, observation_dim, action_dim)
    split_tgts = split_sequence(tgts, observation_dim, action_dim)

    acc = {}
    for (name, pred), (name2, tgt) in zip(
        dict(**split_preds, total=preds).items(),
        dict(**split_tgts, total=tgts).items(),
    ):
        assert name == name2
        acc[f"{name} accuracy"] = pred == tgt

    chunk_acc = {}
    for (name, pred), (name2, tgt) in zip(
        split_preds.items(),
        split_tgts.items(),
    ):
        assert name == name2
        acc[f"{name} accuracy"] = pred == tgt
        _, seq_len, *_ = pred.shape
        chunk_size = seq_len // graphs_per_component
        for i in range(graphs_per_component):
            start = i * chunk_size
            end = (i + 1) * chunk_size

            def get_chunk(x):
                if x.ndim == 2:
                    x = x[..., None]
                return x[:, start:end]

            pred_chunk = get_chunk(pred)
            tgt_chunk = get_chunk(tgt)
            chunk_acc[f"({i}) {name} accuracy"] = pred_chunk == tgt_chunk

    logs = dict(**acc, **chunk_acc)
    return {k: v.float().mean().item() for k, v in logs.items()}
