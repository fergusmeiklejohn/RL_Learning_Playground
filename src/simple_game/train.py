"""Training entrypoint for Atari experiments using Stable-Baselines3."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def select_device(spec: str) -> str:
    if spec == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return spec


def resolve_policy_kwargs(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    resolved: Dict[str, Any] = {}
    for key, value in raw.items():
        if key.endswith("_class") and isinstance(value, str):
            module_name, attr_name = value.rsplit(".", 1)
            module = importlib.import_module(module_name)
            resolved[key] = getattr(module, attr_name)
        else:
            resolved[key] = value
    return resolved


def build_env(
    cfg: Dict[str, Any],
    mode: str,
    *,
    n_envs: Optional[int] = None,
    seed_offset: Optional[int] = None,
    monitor_dir: Optional[str] = None,
) -> VecEnv:
    if seed_offset is None:
        seed_offset = 0 if mode == "train" else 10_000

    env = make_atari_env(
        cfg["environment"]["id"],
        n_envs=n_envs or cfg["model"]["n_envs"],
        seed=cfg["experiment"]["seed"] + seed_offset,
        wrapper_kwargs={
            "noop_max": cfg["environment"].get("noop_max", 30),
            "frame_skip": cfg["environment"].get("frame_skip", 4),
            "terminal_on_life_loss": cfg["environment"].get("terminate_on_life_loss", False),
        },
        monitor_dir=monitor_dir if monitor_dir is not None else cfg["logging"].get("monitor_dir"),
    )
    env = VecFrameStack(env, n_stack=cfg["environment"].get("frame_stack", 4))
    return VecTransposeImage(env)


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in ("tensorboard_log", "checkpoint_dir", "video_dir"):
        path = cfg["logging"].get(key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)


def setup_callbacks(cfg: Dict[str, Any], eval_env: VecEnv) -> list:
    callbacks = []
    if cfg["experiment"].get("save_interval"):
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, cfg["experiment"]["save_interval"] // cfg["model"]["n_envs"]),
                save_path=cfg["logging"]["checkpoint_dir"],
                name_prefix=cfg["experiment"]["name"],
            )
        )
    if cfg["experiment"].get("eval_interval"):
        callbacks.append(
            EvalCallback(
                eval_env,
                eval_freq=max(1, cfg["experiment"]["eval_interval"] // cfg["model"]["n_envs"]),
                n_eval_episodes=cfg["experiment"].get("eval_episodes", 5),
                deterministic=True,
            )
        )
    return callbacks


def record_video(cfg: Dict[str, Any], model: PPO) -> None:
    if not cfg["experiment"].get("rollout_video"):
        return

    video_dir = Path(cfg["logging"]["video_dir"]) / cfg["experiment"]["name"]
    video_dir.mkdir(parents=True, exist_ok=True)

    monitor_path = video_dir / "monitor"
    monitor_path.mkdir(parents=True, exist_ok=True)

    video_env = build_env(
        cfg,
        mode="eval",
        n_envs=1,
        seed_offset=cfg["experiment"].get("video_seed_offset", 2024),
        monitor_dir=str(monitor_path),
    )

    max_steps = cfg["experiment"].get("video_length", 2000)
    prefix = cfg["experiment"].get("video_name", "eval")
    video_env = VecVideoRecorder(
        video_env,
        str(video_dir),
        record_video_trigger=lambda step: step == 0,
        video_length=max_steps,
        name_prefix=prefix,
    )

    obs = video_env.reset()
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = video_env.step(action)
        if dones.any():
            break

    video_env.close()


def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    ensure_dirs(cfg)

    device = select_device(cfg["hardware"].get("device", "auto"))
    if device == "mps" and cfg["hardware"].get("mps_fallback", True):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    train_env = build_env(cfg, mode="train")
    eval_env = build_env(cfg, mode="eval")

    policy_kwargs = resolve_policy_kwargs(cfg["model"].get("policy_kwargs"))

    model = PPO(
        cfg["model"].get("policy", "CnnPolicy"),
        train_env,
        learning_rate=cfg["model"].get("learning_rate", 2.5e-4),
        n_steps=cfg["model"].get("n_steps", 128),
        batch_size=cfg["model"].get("batch_size", 256),
        n_epochs=cfg["model"].get("n_epochs", 4),
        gamma=cfg["model"].get("gamma", 0.99),
        gae_lambda=cfg["model"].get("gae_lambda", 0.95),
        clip_range=cfg["model"].get("clip_range", 0.1),
        ent_coef=cfg["model"].get("ent_coef", 0.01),
        vf_coef=cfg["model"].get("vf_coef", 0.5),
        max_grad_norm=cfg["model"].get("max_grad_norm", 0.5),
        tensorboard_log=cfg["logging"].get("tensorboard_log"),
        verbose=cfg["logging"].get("verbose", 1),
        device=device,
        policy_kwargs=policy_kwargs,
    )

    callbacks = setup_callbacks(cfg, eval_env)
    model.learn(
        total_timesteps=cfg["experiment"]["total_timesteps"],
        callback=callbacks if callbacks else None,
        progress_bar=True,
    )

    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_dir / f"{cfg['experiment']['name']}_final")

    returns, lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=cfg["experiment"].get("eval_episodes", 5),
        deterministic=True,
        return_episode_rewards=True,
    )
    mean_return = sum(returns) / len(returns)
    mean_length = sum(lengths) / len(lengths)
    print(f"[train] Eval mean return {mean_return:.2f} across {len(returns)} episodes")
    print(f"[train] Eval mean length {mean_length:.1f} steps")

    record_video(cfg, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Atari agent with Stable-Baselines3")
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_breakout.yaml"))
    args = parser.parse_args()
    main(args.config)
