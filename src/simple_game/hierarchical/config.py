"""Dataclasses describing the hierarchical control setup."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SkillConfig:
    """Configuration template for a low-level skill policy."""

    name: str
    trigger: str
    horizon: int
    success_reward: float = 1.0
    failure_penalty: float = 0.0
    termination_on_success: bool = True
    warmup_steps: int = 0
    intrinsic: Dict[str, float] = field(default_factory=dict)
    intrinsic_weight: float = 1.0
    epsilon_start: Optional[float] = None
    epsilon_end: Optional[float] = None
    epsilon_decay_steps: Optional[int] = None


@dataclass
class ManagerConfig:
    """Configuration for the high-level option selector."""

    horizon: int
    update_interval: int
    gamma: float = 0.99
    intrinsic_reward_scale: float = 0.0
    advantage_target: str = "delta"
    temperature: float = 1.0


@dataclass
class HierarchicalConfig:
    """Aggregate configuration for hierarchical control training."""

    manager: ManagerConfig
    skills: List[SkillConfig] = field(default_factory=list)
    total_timesteps: int = 3_000_000
    high_level_warmup: int = 50_000
    skill_warmup: int = 10_000
    rollout_stack: int = 4
    frame_skip: int = 4
    eval_interval: int = 100_000
    checkpoint_interval: int = 250_000
    shared_encoder: bool = True
    encoder_dim: int = 256
    device: Optional[str] = None
    buffer_size: int = 100_000
    batch_size: int = 64
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    target_update_interval: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1_000_000
    manager_learning_rate: float = 2.5e-4
    manager_epsilon_start: float = 1.0
    manager_epsilon_end: float = 0.05
    manager_epsilon_decay_steps: int = 1_500_000
    gradient_updates_per_step: int = 1
    eval_games: int = 10
