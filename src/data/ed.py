import data.full_history
from envs.value_iteration import ValueIteration
from pretty import console


class Data(data.full_history.Data):
    def collect_data(self, grid_world: ValueIteration, **kwargs):
        console.log("Value iteration...")
        for t, (V, Pi) in enumerate(
            (grid_world.value_iteration(**kwargs, n_rounds=self.n_rounds))
        ):
            g, s, a, r, d = grid_world.get_trajectories(
                Pi=Pi, n_episodes=self.n_episodes
            )
            console.log(
                f"Round: {t}. Reward: {r.sum(-1).mean().item():.2f}. Value: {V.mean().item():.2f}."
            )
        yield g, s, a, r, d

    @property
    def episodes_per_rollout(self):
        return self.n_episodes
