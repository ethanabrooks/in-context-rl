from dataclasses import asdict, astuple, dataclass

import numpy as np
import torch
from gym.spaces import Discrete, MultiDiscrete, Space
from tqdm import tqdm

from data.base import Data, Step
from envs.parallel.dummy_vec_env import DummyVecEnv
from envs.parallel.subproc_vec_env import SubprocVecEnv
from models import GPT


def get_return(*rewards: float, gamma: float) -> float:
    actual_return = 0
    for r in rewards[::-1]:
        actual_return = r + gamma * actual_return
    return actual_return


def clamp(action: torch.Tensor, space: Space):
    if isinstance(space, Discrete):
        return torch.clamp(action, min=0, max=space.n - 1)
    elif isinstance(space, MultiDiscrete):
        return torch.clamp(action, min=0, max=space.nvec - 1)
    else:
        raise NotImplementedError


def get_metric(
    info,
    rewards,
    gamma,
):
    actual_return = get_return(*rewards, gamma=gamma)
    optimal = info.get("optimal", None)
    if optimal is None:
        return "return", actual_return
    else:
        optimal_return = get_return(*optimal, gamma=gamma)
        regret = optimal_return - actual_return
        return "regret", regret


class Evaluator:
    @classmethod
    def evaluate(cls, dataset: Data, dummy_vec_env: bool, n_rollouts: int, **kwargs):
        N = n_rollouts
        env_fns = [dataset.build_env for _ in range(N)]
        envs: SubprocVecEnv
        envs = DummyVecEnv(env_fns) if dummy_vec_env else SubprocVecEnv(env_fns)
        task = torch.tensor(envs.get_task()).cuda()
        if not dataset.include_goal:
            task = torch.zeros_like(task)
        try:
            evaluator = cls.make_rollout(
                dataset=dataset, envs=envs, n_rollouts=n_rollouts, **kwargs, task=task
            )
            yield from evaluator.rollout()
        finally:
            envs.close()

    @staticmethod
    def make_rollout(*args, **kwargs):
        return Rollout(*args, **kwargs)


@dataclass(frozen=True)
class StepResult:
    reward: np.ndarray
    observation: np.ndarray
    done: np.ndarray
    info: list[dict]


def get_component_idxs(dims: Step):
    cumsum = 0
    step_dim = sum(astuple(dims))
    timestep_start = (
        dims.tasks + dims.observations
    )  # index on which a new timestep starts
    for name, dim in asdict(dims).items():
        pair = np.array([cumsum, cumsum + dim])
        yield name, pair - (cumsum // timestep_start) * step_dim
        cumsum = cumsum + dim


@dataclass
class Rollout:
    dataset: Data
    envs: SubprocVecEnv
    gamma: float
    n_rollouts: int
    net: GPT
    task: torch.Tensor

    def get_action(self, history: torch.Tensor, t: int) -> torch.Tensor:
        action = self.predict_many(history, t, *self.idxs.actions)
        return clamp(action, self.envs.action_space)

    def generate_predictions(self, history: torch.Tensor, t: int, i: int, j: int):
        start = self.index(t)
        for k in range(i, j):
            prediction = self.predict(history, t, k)
            yield prediction
            history[:, start + k] = prediction

    @property
    def idxs(self):
        idxs = dict(get_component_idxs(self.dataset.dims))
        return Step[tuple[int, int]](**idxs)

    def index(self, time_step: int):
        S = self.dataset.step_dim
        W = self.dataset.context_size
        return W + (time_step + 1) * S

    def init_history(self):
        N = self.n_rollouts
        O = self.dataset.dims.observations
        T = self.dataset.episode_length * self.dataset.episodes_per_rollout
        history = torch.full(
            (N, self.index(T)),
            self.dataset.pad_value,
            dtype=torch.long,
            device="cuda",
        )
        start = self.index(-1)
        # observation
        observation = self.envs.reset()
        assert [*observation.shape] == [N, O]
        observation = torch.tensor(observation).cuda(0)
        i, j = start + self.idxs.tasks
        history[:, i:j] = self.task
        i, j = start + self.idxs.observations
        history[:, i:j] = observation
        return history

    def predict(self, history: torch.Tensor, t: int, i: int) -> torch.Tensor:
        dataset = self.dataset
        net = self.net
        N = self.n_rollouts
        K = dataset.n_tokens + 1
        W = self.dataset.context_size

        end = self.index(t)
        ctx = history[:, end - W : end]

        # pass through net
        logits: torch.Tensor
        logits, _ = net.forward(ctx)
        assert [*logits.shape] == [N, W - 1, 1 + dataset.n_tokens]

        # sample probs
        probs = logits[:, i].softmax(dim=-1)
        assert [*probs.shape] == [N, K]
        [prediction] = torch.multinomial(probs, num_samples=1).T
        return prediction

    def predict_many(self, history: torch.Tensor, t: int, start: int, end: int):
        return torch.stack([*self.generate_predictions(history, t, start, end)], dim=-1)

    def reset(self, history: torch.Tensor, n: int, t: int):
        observation = self.envs.reset(n)
        start = self.index(t)
        i, j = start + self.idxs.observations
        history[n, i:j] = observation.cuda()

    def rollout(self):
        A = self.dataset.dims.actions
        N = self.n_rollouts
        O = self.dataset.dims.observations
        T = self.dataset.episode_length * self.dataset.episodes_per_rollout
        dataset = self.dataset

        T = dataset.episode_length * dataset.episodes_per_rollout
        episode_count = np.zeros(N, dtype=int)
        episode_rewards = np.zeros((N, dataset.episode_length))
        episode_t = np.zeros(N, dtype=int)

        history = self.init_history()
        for t in tqdm(range(T)):
            start = self.index(t)
            action = self.get_action(history, t)
            history[:, start - 1 - A : start - 1] = action
            step = self.step(action, history, t)

            assert [*step.observation.shape] == [N, O]
            assert [*step.reward.shape] == [N]
            assert [*step.done.shape] == [N]
            assert len(step.info) == N

            episode_rewards[np.arange(N), episode_t] = step.reward
            episode_t += 1

            for n, (d, ec, er, et, i) in enumerate(
                zip(step.done, episode_count, episode_rewards, episode_t, step.info)
            ):
                assert isinstance(d, (bool, np.bool_))
                assert isinstance(i, dict)
                if d:
                    name, x = get_metric(info=i, rewards=er[:et], gamma=self.gamma)
                    yield dict(n=n, name=name, episode=ec, metric=x)
                    episode_count[n] += 1
                    episode_rewards[n] = 0
                    episode_t[n] = 0
                    self.reset(history, n, t)

    def step(self, action: torch.Tensor, history: torch.Tensor, t: int):
        observation, reward, done, info = self.envs.step(
            action.squeeze(0).cpu().numpy()
        )
        step = StepResult(reward=reward, observation=observation, done=done, info=info)
        start = self.index(t)
        i, _ = start + self.idxs.rewards
        history[:, i] = torch.tensor(step.reward).cuda()
        i, j = start + self.idxs.tasks
        history[:, i:j] = self.task
        i, j = start + self.idxs.observations
        history[:, i:j] = torch.tensor(observation).cuda()
        return step
