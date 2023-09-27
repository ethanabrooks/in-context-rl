import functools
from dataclasses import asdict, astuple, dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiDiscrete, Space
from tqdm import tqdm

from data import Data, Step
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
    elif isinstance(space, Box):
        low = torch.tensor(space.low).cuda()
        high = torch.tensor(space.high).cuda()
        return torch.clamp(action, min=low, max=high)
    else:
        raise NotImplementedError


def get_metric(
    optimal: Optional[Iterable[float]],
    rewards: Iterable[float],
    gamma: float,
):
    actual_return = get_return(*rewards, gamma=gamma)
    if optimal is None:
        return "return", actual_return
    else:
        optimal_return = get_return(*optimal, gamma=gamma)
        regret = optimal_return - actual_return
        assert regret >= 0
        return "regret", regret


class Evaluator:
    def evaluate(
        self,
        dataset: Data,
        dummy_vec_env: bool,
        n_rollouts: int,
        use_heldout_tasks: bool,
        **kwargs
    ):
        N = n_rollouts
        env_fns = [
            functools.partial(
                dataset.build_env, seed=i, use_heldout_tasks=use_heldout_tasks
            )
            for i in range(N)
        ]
        envs: SubprocVecEnv
        envs = DummyVecEnv(env_fns) if dummy_vec_env else SubprocVecEnv(env_fns)
        task = torch.tensor(envs.get_task()).cuda()
        try:
            evaluator = self.make_rollout(
                dataset=dataset,
                envs=envs,
                **kwargs,
                n_rollouts=n_rollouts,
                raw_task=task
            )
            yield from evaluator.rollout()
        finally:
            envs.close()

    def make_rollout(self, *args, **kwargs):
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
    raw_task: torch.Tensor

    @property
    def episode_length(self):
        return self.dataset.episode_length

    @property
    def episodes_per_rollout(self):
        return self.dataset.episodes_per_rollout

    @property
    def task(self):
        if self.dataset.include_task:
            return self.raw_task
        return torch.zeros_like(self.raw_task)

    def get_action(self, history: torch.Tensor, t: int, episode_t: int) -> torch.Tensor:
        action = self.predict_many(history, t, *self.idxs.actions)
        return clamp(action, self.envs.action_space)

    def generate_predictions(self, history: torch.Tensor, t: int, i: int, j: int):
        for k in range(i, j):
            prediction = self.predict(history, t, k)
            yield prediction

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
        T = self.episode_length * self.episodes_per_rollout
        history = torch.full(
            (N, self.index(T)),
            self.dataset.pad_value,
            dtype=torch.float,
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
        S = dataset.step_dim
        W = self.dataset.context_size

        end = self.index(t)
        start = end - W
        ctx = history[:, start:end]

        # pass through net
        logits: torch.Tensor
        logits, _ = net.forward(ctx)
        assert [*logits.shape] == [N, W - 1, 1 + dataset.n_tokens]

        index_history = i + end
        if i > 0:
            index_history -= S
        index_logits = index_history - start  # subtract start of history
        index_logits = index_logits - 1  # subtract index offset

        pad_value = self.dataset.pad_value
        assert torch.any(ctx[:, index_logits] != pad_value)
        assert torch.any(history[:, index_history - 1] != pad_value)
        assert torch.all(ctx[:, index_logits + 1] == pad_value)
        assert torch.all(history[:, index_history] == pad_value)

        # sample probs
        [prediction] = self.net.predict(logits[:, index_logits]).T
        history[:, index_history] = prediction
        return prediction

    def predict_many(self, history: torch.Tensor, t: int, start: int, end: int):
        return torch.stack([*self.generate_predictions(history, t, start, end)], dim=-1)

    def reset(self, history: torch.Tensor, n: int, t: int):
        observation = self.envs.reset(n)
        start = self.index(t)
        i, j = start + self.idxs.observations
        history[n, i:j] = torch.tensor(observation).cuda()

    def rollout(self):
        A = self.dataset.dims.actions
        N = self.n_rollouts
        O = self.dataset.dims.observations
        T = self.episode_length * self.episodes_per_rollout

        episode_count = np.zeros(N, dtype=int)
        actions = np.zeros((T, N, A))
        observations = np.zeros((T, N, O))
        rewards = np.zeros((T, N))
        terminations = np.zeros((T, N), dtype=int)
        episode_timesteps = np.zeros(N, dtype=int)
        history = self.init_history()
        for t in tqdm(range(T)):
            start = self.index(t)
            action = self.get_action(history, t, episode_timesteps)
            actions[t] = action.cpu().numpy()
            history[:, start - 1 - A : start - 1] = action
            step = self.step(action, history, t)

            assert len(step.info) == N
            observations[t] = step.observation
            rewards[t] = step.reward
            terminations[t] = step.done

            episode_timesteps += 1

            for n, (d, i) in enumerate(zip(step.done, step.info)):
                assert isinstance(d, (bool, np.bool_))
                assert isinstance(i, dict)
                optimal = i.get("optimal", None)
                if d:
                    episode_timestep = episode_timesteps[n]
                    t0 = max(0, t + 1 - episode_timestep)
                    t1 = t + 1
                    _, x = get_metric(
                        optimal=optimal,
                        rewards=rewards[t0:t1, n],
                        gamma=self.gamma,
                    )
                    yield dict(
                        n=n,
                        states=observations[t0:t1, n],
                        actions=actions[t0:t1, n],
                        rewards=rewards[t0:t1, n],
                        episode=episode_count[n],
                        metric=x,
                        task=self.raw_task[n].cpu().numpy(),
                    )
                    episode_count[n] += 1
                    episode_timesteps[n] = 0
                    self.reset(history, n, t)

    def step(self, action: torch.Tensor, history: torch.Tensor, t: int) -> StepResult:
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
