from abc import ABC, abstractmethod

import gym


class Env(gym.Env, ABC):
    @abstractmethod
    def get_task(self):
        pass

    @property
    @abstractmethod
    def observation_space(self):  # dead: disable
        pass

    @property
    @abstractmethod
    def action_space(self):
        pass
