import itertools

import numpy as np
from tqdm import tqdm

import point_env.learned
from point_env.env import PointEnv


def generate_synthetic_data(
    env: PointEnv,
    max_steps: int,
    n_histories: int,
):
    for _ in range(n_histories):
        goal = env.reset_task()
        state = env.reset()
        done_mdp = False
        steps = 0
        while not done_mdp and steps < max_steps:
            action = (goal - state) / 0.1
            next_state, reward, done_mdp, _ = env.step(action)
            done = done_mdp
            yield dict(
                state=state,
                actions=action,
                next_state=next_state,
                rewards=reward,
                done=done,
                done_mdp=done_mdp,
                task=goal,
            )

            state = next_state
            steps += 1


class Data(point_env.learned.Data):
    def __init__(
        self,
        *args,
        max_steps: int,
        n_histories: int,
        **kwargs,
    ):
        self.max_steps = max_steps
        self.n_histories = n_histories
        super().__init__(*args, omit_episodes=None, **kwargs)

    def get_data(self):
        env = self.build_env(
            seed=0,
            use_heldout_tasks=False,
            include_optimal=False,
            max_episode_steps=self.max_steps,
        )
        dtype = [
            ("task", "<f4", (2,)),
            ("actions", "<f4", (2,)),
            ("rewards", "<f4", (1,)),
            ("done", "<f8", (1,)),
            ("state", "<f4", (2,)),
            ("done_mdp", "?", (1,)),
            ("next_state", "<f4", (2,)),
        ]
        n_data = self.max_steps * self.n_histories
        array = np.empty(n_data, dtype=dtype)

        data = tqdm(
            itertools.islice(
                generate_synthetic_data(
                    env=env,
                    max_steps=self.max_steps,
                    n_histories=self.n_histories,
                ),
                n_data,
            ),
            desc="Generating synthetic data",
            total=n_data,
        )

        for i, entry in enumerate(data):
            row = [entry[k] for k, _, _ in dtype]
            array[i] = tuple(row)
        return array
