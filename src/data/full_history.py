import torch
import torch.nn.functional as F

import data.base
from pretty import console
from tabular.value_iteration import ValueIteration


def expand_as(x: torch.Tensor, y: torch.Tensor):
    while x.dim() < y.dim():
        x = x[..., None]
    return x.expand_as(y)


class Data(data.base.Data):
    def __init__(
        self,
        grid_size: int,
        grid_world_args: dict,
        include_goal: bool,
        n_data: int,
        optimal_policy: bool,
        steps_per_context: int,
        value_iteration_args: dict,
    ):
        self.n_data = n_data
        self.steps_per_context = steps_per_context
        episode_length = 1 + grid_size * 2
        grid_world = ValueIteration(
            **grid_world_args, grid_size=grid_size, n_tasks=n_data
        )
        n_rounds = 2 * grid_size - 1

        def collect_data():
            console.log("Value iteration...")
            for t, (V, Pi) in enumerate(
                (grid_world.value_iteration(**value_iteration_args, n_rounds=n_rounds))
            ):
                g, s, a, r, d = grid_world.get_trajectories(
                    Pi=Pi, episode_length=episode_length
                )
                console.log(
                    f"Round: {t}. Reward: {r.sum(-1).mean().item():.2f}. Value: {V.mean().item():.2f}."
                )
                yield g, s, a, r, d

        data = list(collect_data())
        if optimal_policy:
            data = data[-1:]
        components = zip(*data)
        components = [torch.cat(c, dim=1) for c in components]
        (
            self.goals,
            self.observations,
            self.actions,
            self.rewards,
            self.done,
        ) = components
        if not include_goal:
            self.goals = torch.zeros_like(self.goals)
        components = [self.goals, self.observations, self.actions, self.rewards]
        masks = [
            expand_as(~self.done, c).roll(dims=[1], shifts=[1])
            for c in [self.goals, self.observations]
        ] + [torch.ones_like(c) for c in [self.actions, self.rewards]]
        mask = self.cat_sequence(*masks)
        data = self.cat_sequence(*components)
        sequence = self.split_sequence(data)
        for name, component in sequence.items():
            assert (getattr(self, name) == component).all()

        data = data.reshape(n_data, -1, self.step_dim)
        mask = mask.reshape(n_data, -1, self.step_dim)
        _, self.steps_per_row, _ = data.shape
        self.unpadded_data = data
        assert [*self.unpadded_data.shape] == [
            n_data,
            self.steps_per_row,
            self.step_dim,
        ]
        pad_value = self.unpadded_data.max().item() + 1
        self.data = F.pad(data, (0, 0, steps_per_context, 0), value=pad_value).cuda()
        self.mask = F.pad(mask, (0, 0, steps_per_context, 0), value=0).cuda()

    def __getitem__(self, idx):
        i, j = self.index_1d_to_2d(idx)
        jj = slice(j, j + self.steps_per_context)
        return self.data[i, jj].view(-1), self.mask[i, jj].view(-1)

    def __len__(self):
        return self.n_data * self.steps_per_row

    @property
    def _dims(self):
        _, _, goal_dim = self.goals.shape
        _, _, obs_dim = self.observations.shape
        return [goal_dim, obs_dim, 1, 1]

    def cat_sequence(self, goals, observations, actions, rewards):
        data = torch.cat(
            [goals, observations, actions[..., None], rewards[..., None]],
            dim=-1,
        )
        n_data, _, _ = data.shape
        return data.long().reshape(n_data, -1).contiguous()

    def index_1d_to_2d(self, index):
        row = index // self.steps_per_row
        col = index % self.steps_per_row
        return (row, col)

    def split_sequence(self, sequence: torch.Tensor):
        n_batch, _ = sequence.shape
        sequence = sequence.reshape(n_batch, -1, self.step_dim)
        goals, observations, actions, rewards = sequence.split(self._dims, dim=-1)
        actions = actions.squeeze(-1)
        rewards = rewards.squeeze(-1)
        return dict(
            goals=goals, observations=observations, actions=actions, rewards=rewards
        )

    def get_metrics(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        sequence: torch.Tensor,
        steps_per_graph: int,
    ):
        n_batch, seq_len = sequence.shape
        n_batch2, seq_len2, _ = logits.shape
        assert n_batch == n_batch2
        assert seq_len == seq_len2 + 1

        prefix = sequence[:, :1]
        preds = torch.cat([prefix, logits.argmax(-1)], dim=1)
        tgts = sequence
        split_preds = self.split_sequence(preds)
        split_tgts = self.split_sequence(tgts)
        split_masks = self.split_sequence(mask)

        acc = {}
        acc["(total) accuracy"] = (preds == tgts)[mask.bool()]
        iterator = list(
            zip(
                split_preds.items(),
                split_tgts.items(),
                split_masks.items(),
            )
        )
        for (name, pred), (name2, tgt), (name3, mask) in iterator:
            assert name == name2 == name3
            total_accuracy = (pred == tgt)[mask.bool()]
            acc[f"(total) {name} accuracy"] = total_accuracy

        for i in range(seq_len):
            for (name, pred), (name2, tgt), (name3, mask) in iterator:
                assert name == name2 == name3
                _, component_seq_len, *_ = pred[mask].shape
                graphs_per_component = component_seq_len // steps_per_graph

                if i >= graphs_per_component:
                    continue

                def get_chunk(x, start, end):
                    if x.ndim == 2:
                        x = x[..., None]
                    return x[:, start:end]

                start = i * steps_per_graph
                end = (i + 1) * steps_per_graph

                pred_chunk = get_chunk(pred, start, end)
                tgt_chunk = get_chunk(tgt, start, end)
                mask_chunk = get_chunk(mask, start, end)
                if mask_chunk.sum() > 0:
                    acc[f"({i}) {name} accuracy"] = (pred_chunk == tgt_chunk)[
                        mask_chunk.bool()
                    ]

        return {k: v.float().mean().item() for k, v in acc.items()}
