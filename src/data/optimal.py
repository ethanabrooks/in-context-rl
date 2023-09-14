import data.full_history


class Data(data.full_history.Data):
    @property
    def episodes_per_rollout(self):
        return self.n_episodes
