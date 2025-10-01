"""Custom algorithms built on top of Stable-Baselines3."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import CnnPolicy, QNetwork
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import Schedule
from gymnasium import spaces

from .prioritized_buffer import (
    PrioritizedDictReplayBuffer,
    PrioritizedReplayBuffer,
    PrioritizedReplayBufferSamples,
)


class DuelingQNetwork(QNetwork):
    """Q-network with dueling value and advantage streams."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )
        action_dim = int(action_space.n)

        advantage_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        value_net = create_mlp(self.features_dim, 1, self.net_arch, self.activation_fn)
        self.advantage_net = nn.Sequential(*advantage_net)
        self.value_net = nn.Sequential(*value_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        advantages = self.advantage_net(features)
        values = self.value_net(features)
        advantages = advantages - advantages.mean(dim=1, keepdim=True)
        return values + advantages


class DuelingCnnPolicy(CnnPolicy):
    """CNN policy that builds a dueling Q-network."""

    def make_q_net(self) -> DuelingQNetwork:  # type: ignore[override]
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)



class PrioritizedDQN(DQN):
    """DQN with optional prioritized experience replay support."""

    def __init__(
        self,
        *args: Any,
        prioritized_replay: bool = False,
        prioritized_replay_alpha: float = 0.6,
        prioritized_replay_beta0: float = 0.4,
        prioritized_replay_beta_iters: Optional[int] = None,
        prioritized_replay_eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.use_prioritized = prioritized_replay
        self.prioritized_eps = prioritized_replay_eps

        if self.use_prioritized:
            env = None
            if len(args) > 1:
                env = args[1]
            elif "env" in kwargs:
                env = kwargs["env"]
            obs_space = None
            if env is not None:
                obs_space = env.observation_space
            replay_buffer_kwargs: Dict[str, Any] = kwargs.pop("replay_buffer_kwargs", {}) or {}
            replay_buffer_kwargs.update(
                {
                    "alpha": prioritized_replay_alpha,
                    "beta0": prioritized_replay_beta0,
                    "beta_iters": prioritized_replay_beta_iters,
                    "eps": prioritized_replay_eps,
                }
            )
            if obs_space is not None and isinstance(obs_space, spaces.Dict):
                kwargs.setdefault("replay_buffer_class", PrioritizedDictReplayBuffer)
            else:
                kwargs.setdefault("replay_buffer_class", PrioritizedReplayBuffer)
            kwargs.setdefault("replay_buffer_kwargs", replay_buffer_kwargs)

        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:  # type: ignore[override]
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if isinstance(replay_data, PrioritizedReplayBufferSamples):
                weights = replay_data.weights.unsqueeze(1)
                batch_indices = replay_data.indices.cpu().numpy()
            else:
                weights = None
                batch_indices = None

            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            elementwise_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
            if weights is not None:
                loss = (elementwise_loss * weights).mean()
            else:
                loss = elementwise_loss.mean()
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            if self.use_prioritized and batch_indices is not None:
                td_errors = (target_q_values - current_q_values).detach().abs().cpu().numpy().flatten()
                self.replay_buffer.update_priorities(batch_indices, td_errors + self.prioritized_eps)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


__all__ = ["PrioritizedDQN", "DuelingCnnPolicy"]
