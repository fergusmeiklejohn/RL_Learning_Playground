"""Training scaffolding for hierarchical controllers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch as th
from gymnasium.vector import VectorEnv

from ..train import build_env, load_config, ensure_dirs, resolve_policy_kwargs
from ..evaluate import evaluate_checkpoint  # type: ignore
from .config import HierarchicalConfig, ManagerConfig, SkillConfig
from .controller import HierarchicalController


@dataclass
class TrainerArtifacts:
    """Paths and handles produced during hierarchical training."""

    checkpoint_path: Optional[str]
    manager_path: Optional[str]
    skill_paths: Dict[str, str]


class HierarchicalTrainer:
    """Placeholder trainer coordinating manager and skill optimisation."""

    def __init__(self, cfg: HierarchicalConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device or ("mps" if th.backends.mps.is_available() else "cpu")
        self.controller: Optional[HierarchicalController] = None

    def build_controller(self, env: VectorEnv) -> HierarchicalController:
        controller = HierarchicalController(env.single_observation_space, env.single_action_space, self.cfg, self.device)
        self.controller = controller
        return controller

    def train(self) -> TrainerArtifacts:
        raise NotImplementedError("HierarchicalTrainer.train is pending implementation")

