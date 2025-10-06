"""Dataclasses describing the hierarchical control setup."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


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

