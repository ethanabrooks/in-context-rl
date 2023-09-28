import numpy as np
from gym import spaces

from envs.base import Env


def point_on_circle(angle: float, radius: float = 1.0):
    return radius * np.array((np.cos(angle), np.sin(angle)))


class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(
        self,
        seed: int,
        goal_sampler: str = None,
        optimal: list[float] = None,
        test: bool = False,
        test_threshold: float = np.pi / 2,
    ):
        self.random = np.random.default_rng(seed)
        goal_samplers = {
            "semi-circle": self.sample_semi_circle,
            "circle": self.sample_circle,
            "double-arc": self.sample_double_arc,
        }
        self.goal_sampler = sampler = goal_samplers[goal_sampler]
        if goal_sampler == "double-arc":
            self.goal_sampler = lambda: (
                sampler(0, test_threshold)
                if not test
                else sampler(test_threshold, np.pi)
            )

        self.optimal = optimal
        self.reset_task()
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

        points = np.stack([point_on_circle(a) for a in np.linspace(0, 2 * np.pi, 4)])
        self._task_space = spaces.Box(
            low=points.min(axis=0), high=points.max(axis=0), shape=(2,)
        )
        self._optimal_rewards_iterator = None
        self._optimal = None

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def task_space(self):
        return self._task_space

    def _get_obs(self):
        return np.copy(self._state)

    def get_task(self):
        return self._goal

    def optimal_rewards_generator(
        self,
        start_state: np.ndarray,
        goal: np.ndarray,
    ):
        current_state = start_state

        while True:
            optimal_action = goal - current_state
            optimal_action = np.clip(
                optimal_action, self.action_space.low, self.action_space.high
            )
            current_state = current_state + optimal_action
            yield -np.linalg.norm(current_state - goal, ord=2)

    def reset(self):
        return self.reset_model()

    def reset_model(self):
        self._state = np.zeros(2)
        self._optimal = []
        self._optimal_rewards_iterator = self.optimal_rewards_generator(
            self._state, self._goal
        )
        return self._get_obs()

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        return task

    def reward(self, state: np.ndarray, goal: np.ndarray):
        return -np.linalg.norm(state - goal, ord=2)

    def sample_circle(self):
        angle = self.random.uniform(0, 2 * np.pi)
        return point_on_circle(angle)

    def sample_double_arc(self, start, end):
        angle = self.random.uniform(start, end)
        angle += self.random.choice(2) * np.pi
        return point_on_circle(angle)

    def sample_semi_circle(self):
        angle = self.random.uniform(0, np.pi)
        return point_on_circle(angle)

    def sample_task(self):
        goal = self.goal_sampler()
        return goal

    def set_task(self, task):
        self._goal = task

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action), action

        self._state = self._state + 0.1 * action
        reward = self.reward(self._state, self._goal)
        done = False
        ob = self._get_obs()
        optimal_reward = next(self._optimal_rewards_iterator)
        self._optimal.append(optimal_reward)
        info = {"task": self.get_task(), "optimal": self._optimal}
        return ob, reward, done, info
