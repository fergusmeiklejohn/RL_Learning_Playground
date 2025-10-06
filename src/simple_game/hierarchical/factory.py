"""Factory helpers to build hierarchical configs from raw dictionaries."""
from __future__ import annotations

from typing import Any, Dict

from .config import HierarchicalConfig, ManagerConfig, SkillConfig


def build_config(raw: Dict[str, Any]) -> HierarchicalConfig:
    """Construct a HierarchicalConfig from raw YAML data."""

    manager_data = raw.get("manager") or {}
    manager = ManagerConfig(
        horizon=int(manager_data.get("horizon", 32)),
        update_interval=int(manager_data.get("update_interval", 8)),
        gamma=float(manager_data.get("gamma", 0.99)),
        intrinsic_reward_scale=float(manager_data.get("intrinsic_reward_scale", 0.0)),
        advantage_target=str(manager_data.get("advantage_target", "delta")),
        temperature=float(manager_data.get("temperature", 1.0)),
    )

    skills = []
    for entry in raw.get("skills", []):
        skills.append(
            SkillConfig(
                name=str(entry.get("name")),
                trigger=str(entry.get("trigger", "")),
                horizon=int(entry.get("horizon", manager.update_interval)),
                success_reward=float(entry.get("success_reward", 1.0)),
                failure_penalty=float(entry.get("failure_penalty", 0.0)),
                termination_on_success=bool(entry.get("termination_on_success", True)),
                warmup_steps=int(entry.get("warmup_steps", 0)),
            )
        )

    training = raw.get("training") or {}

    return HierarchicalConfig(
        manager=manager,
        skills=skills,
        total_timesteps=int(raw.get("total_timesteps", 3_000_000)),
        high_level_warmup=int(raw.get("high_level_warmup", 50_000)),
        skill_warmup=int(raw.get("skill_warmup", 10_000)),
        rollout_stack=int(raw.get("rollout_stack", 4)),
        frame_skip=int(raw.get("frame_skip", 4)),
        eval_interval=int(raw.get("eval_interval", 100_000)),
        checkpoint_interval=int(raw.get("checkpoint_interval", 250_000)),
        shared_encoder=bool(raw.get("shared_encoder", True)),
        encoder_dim=int(raw.get("encoder_dim", 256)),
        device=raw.get("device"),
        buffer_size=int(training.get("buffer_size", 100_000)),
        batch_size=int(training.get("batch_size", 64)),
        learning_rate=float(training.get("learning_rate", 2.5e-4)),
        gamma=float(training.get("gamma", 0.99)),
        target_update_interval=int(training.get("target_update_interval", 2_000)),
        epsilon_start=float(training.get("epsilon_start", 1.0)),
        epsilon_end=float(training.get("epsilon_end", 0.05)),
        epsilon_decay_steps=int(training.get("epsilon_decay_steps", 1_000_000)),
        manager_learning_rate=float(training.get("manager_learning_rate", training.get("learning_rate", 2.5e-4))),
        manager_epsilon_start=float(training.get("manager_epsilon_start", training.get("epsilon_start", 1.0))),
        manager_epsilon_end=float(training.get("manager_epsilon_end", training.get("epsilon_end", 0.05))),
        manager_epsilon_decay_steps=int(training.get("manager_epsilon_decay_steps", 1_500_000)),
        gradient_updates_per_step=int(training.get("gradient_updates_per_step", 1)),
        eval_games=int(training.get("eval_games", 10)),
    )
