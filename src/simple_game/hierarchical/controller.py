"""Hierarchical controller scaffolding for option-based agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch as th
import torch.nn as nn

from gymnasium import spaces

from ..policies import SimpleNatureCNN
from .config import HierarchicalConfig, SkillConfig


@dataclass
class OptionState:
    """Runtime bookkeeping for an instantiated option."""

    step_count: int = 0
    terminated: bool = False
    reward_accumulator: float = 0.0
    start_observation: Optional[Any] = None
    option_index: Optional[int] = None


class SkillPolicy(nn.Module):
    """Low-level controller placeholder to be specialised per skill."""

    def __init__(self, observation_space: spaces.Space, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("SkillPolicy expects a Box observation space")
        self.encoder = SimpleNatureCNN(observation_space, features_dim=hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.encoder(obs)
        return self.policy_head(features)

    def act(self, obs: th.Tensor) -> th.Tensor:
        logits = self.forward(obs)
        return logits.argmax(dim=-1)


class ManagerPolicy(nn.Module):
    """High-level policy selecting among available skills."""

    def __init__(self, observation_space: spaces.Space, num_skills: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("ManagerPolicy expects a Box observation space")
        self.encoder = SimpleNatureCNN(observation_space, features_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, num_skills)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.encoder(obs)
        return self.head(features)

    def act(self, obs: th.Tensor, temperature: float = 1.0) -> th.Tensor:
        logits = self.forward(obs)
        if temperature <= 0:
            return logits.argmax(dim=-1)
        dist = th.distributions.Categorical(logits=logits / temperature)
        return dist.sample()


class HierarchicalController:
    """Coordinates a high-level manager with multiple skill policies."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        config: HierarchicalConfig,
        device: Optional[str] = None,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.device = device or ("mps" if th.backends.mps.is_available() else "cpu")
        self.manager = ManagerPolicy(observation_space, len(config.skills), hidden_dim=config.encoder_dim).to(self.device)
        self.skills: Dict[str, SkillPolicy] = {}
        self.skill_specs: Dict[str, SkillConfig] = {spec.name: spec for spec in config.skills}
        for skill_cfg in config.skills:
            skill = SkillPolicy(observation_space, action_space.n, hidden_dim=config.encoder_dim)
            self.skills[skill_cfg.name] = skill.to(self.device)
        self.active_option: Optional[str] = None
        self.option_state = OptionState()
        self.manager_hidden: Optional[th.Tensor] = None

    def reset_option(self, skill_name: str, *, start_obs: Optional[Any] = None, option_index: Optional[int] = None) -> None:
        self.active_option = skill_name
        self.option_state = OptionState(start_observation=start_obs, option_index=option_index)

    def select_option(self, obs: th.Tensor) -> str:
        obs = obs.to(self.device)
        option_idx = self.manager.act(obs.unsqueeze(0), temperature=self.config.manager.temperature)
        option_name = self.config.skills[int(option_idx.item())].name
        self.reset_option(option_name, start_obs=None, option_index=int(option_idx.item()))
        return option_name

    def act(self, obs: th.Tensor) -> int:
        if self.active_option is None or self.option_state.terminated:
            self.select_option(obs)
        assert self.active_option is not None
        skill = self.skills[self.active_option]
        action = skill.act(obs.to(self.device).unsqueeze(0))
        self.option_state.step_count += 1
        return int(action.item())

    def update(self, reward: float, done: bool) -> None:
        self.option_state.reward_accumulator += reward
        if done or (
            self.active_option is not None
            and self.option_state.step_count >= self.skill_specs[self.active_option].horizon
        ):
            self.option_state.terminated = True

    def skill_config(self, name: str) -> SkillConfig:
        return self.skill_specs[name]

    def skill_names(self) -> list[str]:
        return list(self.skill_specs.keys())
