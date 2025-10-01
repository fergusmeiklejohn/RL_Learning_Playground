"""Prioritized replay buffer compatible with Stable-Baselines3 DQN."""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import (
    DictReplayBuffer,
    DictReplayBufferSamples,
    ReplayBuffer,
    ReplayBufferSamples,
)


class PrioritizedReplayBufferSamples(NamedTuple):
    """Replay samples augmented with importance weights and indices."""

    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    weights: th.Tensor
    indices: th.Tensor


class PrioritizedReplayBuffer(ReplayBuffer):
    """Simple proportional prioritized replay buffer.

    Implementation follows the standard approach from Schaul et al. (2015),
    without segment trees (uses numpy cumulative probabilities instead).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: str = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha: float = 0.6,
        beta0: float = 0.4,
        beta_iters: Optional[int] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.alpha = alpha
        self.beta0 = beta0
        self.beta_iters = beta_iters
        self.eps = eps

        self.max_priority = 1.0
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)
        self.sample_count = 0

    def _beta(self) -> float:
        if not self.beta_iters or self.beta_iters <= 0:
            return self.beta0
        progress = min(1.0, self.sample_count / float(self.beta_iters))
        return self.beta0 + progress * (1.0 - self.beta0)

    def add(self, *args, **kwargs) -> None:
        super().add(*args, **kwargs)
        idx = (self.pos - 1) % self.buffer_size
        self.priorities[idx] = self.max_priority

    def _valid_size(self) -> int:
        return self.buffer_size if self.full else self.pos

    def _sample_proportional(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        valid_size = self._valid_size()
        if valid_size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        priorities = self.priorities[:valid_size]
        if priorities.max() == 0:
            priorities = np.ones_like(priorities)

        scaled = priorities ** self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(valid_size, size=batch_size, p=probs)
        return indices, probs[indices]

    def sample(self, batch_size: int, env=None) -> PrioritizedReplayBufferSamples:  # type: ignore[override]
        indices, sample_probs = self._sample_proportional(batch_size)
        beta = self._beta()
        self.sample_count += batch_size

        valid_size = self._valid_size()
        weights = (valid_size * sample_probs) ** (-beta)
        weights /= weights.max()

        samples = self._get_samples(indices, env)

        weights_tensor = th.as_tensor(weights, device=self.device, dtype=samples.rewards.dtype)
        index_tensor = th.as_tensor(indices, device=self.device, dtype=th.long)
        return PrioritizedReplayBufferSamples(
            observations=samples.observations,
            actions=samples.actions,
            next_observations=samples.next_observations,
            dones=samples.dones,
            rewards=samples.rewards,
            weights=weights_tensor,
            indices=index_tensor,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.asarray(priorities, dtype=np.float32)
        updated = np.abs(priorities) + self.eps
        self.priorities[indices] = updated
        self.max_priority = max(self.max_priority, updated.max())


class PrioritizedDictReplayBuffer(DictReplayBuffer):
    """Prioritized replay buffer variant for dict observations."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space,
        device: str = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha: float = 0.6,
        beta0: float = 0.4,
        beta_iters: Optional[int] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        self.alpha = alpha
        self.beta0 = beta0
        self.beta_iters = beta_iters
        self.eps = eps

        self.max_priority = 1.0
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)
        self.sample_count = 0

    def _beta(self) -> float:
        if not self.beta_iters or self.beta_iters <= 0:
            return self.beta0
        progress = min(1.0, self.sample_count / float(self.beta_iters))
        return self.beta0 + progress * (1.0 - self.beta0)

    def add(self, *args, **kwargs) -> None:  # type: ignore[override]
        super().add(*args, **kwargs)
        idx = (self.pos - 1) % self.buffer_size
        self.priorities[idx] = self.max_priority

    def _valid_size(self) -> int:
        return self.buffer_size if self.full else self.pos

    def _sample_proportional(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        valid_size = self._valid_size()
        if valid_size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        priorities = self.priorities[:valid_size]
        if priorities.max() == 0:
            priorities = np.ones_like(priorities)

        scaled = priorities ** self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(valid_size, size=batch_size, p=probs)
        return indices, probs[indices]

    def sample(self, batch_size: int, env=None) -> PrioritizedReplayBufferSamples:  # type: ignore[override]
        indices, sample_probs = self._sample_proportional(batch_size)
        beta = self._beta()
        self.sample_count += batch_size

        valid_size = self._valid_size()
        weights = (valid_size * sample_probs) ** (-beta)
        weights /= weights.max()

        samples: DictReplayBufferSamples = self._get_samples(indices, env)

        weights_tensor = th.as_tensor(weights, device=self.device, dtype=samples.rewards.dtype)
        index_tensor = th.as_tensor(indices, device=self.device, dtype=th.long)
        return PrioritizedReplayBufferSamples(
            observations=samples.observations,
            actions=samples.actions,
            next_observations=samples.next_observations,
            dones=samples.dones,
            rewards=samples.rewards,
            weights=weights_tensor,
            indices=index_tensor,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.asarray(priorities, dtype=np.float32)
        updated = np.abs(priorities) + self.eps
        self.priorities[indices] = updated
        self.max_priority = max(self.max_priority, updated.max())


__all__ = ["PrioritizedReplayBuffer", "PrioritizedDictReplayBuffer", "PrioritizedReplayBufferSamples"]
