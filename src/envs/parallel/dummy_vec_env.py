from multiprocessing.connection import Connection
from typing import Any

from envs.base import Env
from envs.parallel.subproc_vec_env import Command, SubprocVecEnv, work


class DummyRemote(Connection):
    def __init__(self, env: Env):
        self.env = env
        self.storage = None

    def recv(self):
        return self.storage

    def send(self, data: tuple[str, Any]):
        cmd, data = data
        if cmd != Command.CLOSE:
            self.storage = work(self.env, cmd, data)


class DummyVecEnv(SubprocVecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    def start_processes(self, env_fns):
        return [DummyRemote(fn()) for fn in env_fns], []
