"""Training entrypoint for Atari experiments using Stable-Baselines3."""
from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class ProgressConsoleCallback(BaseCallback):
    """Prints periodic progress updates to stdout for shell visibility."""

    def __init__(self, total_timesteps: int, log_interval: int = 10_000) -> None:
        super().__init__()
        self.total_timesteps = max(1, total_timesteps)
        self.log_interval = max(1, log_interval)

    def _on_step(self) -> bool:
        if self.n_calls == 1 or self.num_timesteps % self.log_interval == 0:
            pct = 100.0 * self.num_timesteps / self.total_timesteps
            fps = self.logger.name_to_value.get("time/fps", 0.0)
            reward = self.logger.name_to_value.get("rollout/ep_rew_mean", float("nan"))
            length = self.logger.name_to_value.get("rollout/ep_len_mean", float("nan"))
            print(
                f"[train] {self.num_timesteps:,}/{self.total_timesteps:,} steps"
                f" ({pct:5.1f}%) | fps={fps:.0f} | reward={reward:.1f} | length={length:.0f}",
                flush=True,
            )
        return True


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def select_device(spec: str, enable_fallback: bool = True) -> str:
    """Resolve the torch device to target based on availability."""
    if spec == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = spec

    if device == "mps" and not torch.backends.mps.is_available():
        if not enable_fallback:
            raise RuntimeError("MPS requested but not available on this machine")
        print("[train] MPS unavailable, falling back to CPU")
        device = "cpu"

    return device


def resolve_vec_env(cfg: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    cls_name = cfg["model"].get("vec_env_class", "dummy").lower()
    if cls_name == "subproc":
        vec_env_cls = SubprocVecEnv
        vec_env_kwargs: Dict[str, Any] = {}
        start_method = cfg["model"].get("subproc_start_method")
        if start_method:
            vec_env_kwargs["start_method"] = start_method
    else:
        vec_env_cls = DummyVecEnv
        vec_env_kwargs = {}
    return vec_env_cls, vec_env_kwargs


def build_env(cfg: Dict[str, Any], *, mode: str = "train") -> VecEnv:
    env_id = cfg["environment"]["id"]
    env_seed = cfg["experiment"]["seed"] + (1000 if mode == "eval" else 0)
    n_envs = cfg["model"]["n_envs"]

    wrapper_kwargs = {
        "noop_max": cfg["environment"].get("noop_max", 30),
        "frame_skip": cfg["environment"].get("frame_skip", 4),
        "terminal_on_life_loss": cfg["environment"].get("terminate_on_life_loss", False),
    }

    render_mode = cfg["environment"].get("render_mode")
    env_kwargs = {"render_mode": render_mode} if render_mode else None

    monitor_dir = Path(cfg["logging"].get("monitor_dir", "runs/monitor")) / mode / cfg["experiment"]["name"]
    monitor_dir.mkdir(parents=True, exist_ok=True)

    vec_env_cls, vec_env_kwargs = resolve_vec_env(cfg)

    env = make_atari_env(
        env_id,
        n_envs=n_envs,
        seed=env_seed,
        monitor_dir=str(monitor_dir),
        env_kwargs=env_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )
    env = VecFrameStack(env, n_stack=cfg["environment"].get("frame_stack", 4))
    return env


def build_callbacks(cfg: Dict[str, Any], eval_env: VecEnv, total_timesteps: int) -> List:
    callbacks: List = []
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_interval = cfg["experiment"].get("save_interval")
    if save_interval:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, save_interval // cfg["model"]["n_envs"]),
                save_path=str(checkpoint_dir),
                name_prefix=cfg["experiment"]["name"],
                save_replay_buffer=cfg["logging"].get("save_replay_buffer", False),
            )
        )

    eval_interval = cfg["experiment"].get("eval_interval")
    if eval_interval:
        callbacks.append(
            EvalCallback(
                eval_env,
                eval_freq=max(1, eval_interval // cfg["model"]["n_envs"]),
                n_eval_episodes=cfg["experiment"].get("eval_episodes", 5),
                best_model_save_path=str(checkpoint_dir / "best"),
                deterministic=True,
                render=False,
            )
        )

    console_interval = cfg["logging"].get("console_interval")
    if console_interval is not None:
        callbacks.append(ProgressConsoleCallback(total_timesteps, console_interval))

    return callbacks


def record_eval_video(cfg: Dict[str, Any], model: PPO) -> None:
    if not cfg["experiment"].get("rollout_video", False):
        return

    video_dir = Path(cfg["logging"]["video_dir"]) / cfg["experiment"]["name"]
    video_dir.mkdir(parents=True, exist_ok=True)

    env_id = cfg["environment"]["id"]
    max_steps = cfg["experiment"].get("video_length", 2000)

    # Record a single deterministic episode for qualitative review.
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=str(video_dir), name_prefix="eval")

    obs, info = env.reset(seed=cfg["experiment"]["seed"] + 2024)
    done = False
    step = 0
    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1

    env.close()


def main(config_path: Path) -> None:
    cfg = load_config(config_path)

    run_name = cfg["experiment"]["name"]
    tensorboard_dir = Path(cfg["logging"]["tensorboard_log"]) / run_name
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["logging"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logging"]["video_dir"]).mkdir(parents=True, exist_ok=True)

    device = select_device(cfg["hardware"].get("device", "auto"), cfg["hardware"].get("mps_fallback", True))
    if device == "mps" and cfg["hardware"].get("mps_fallback", True):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    env = build_env(cfg, mode="train")
    eval_env = build_env(cfg, mode="eval")

    model = PPO(
        policy=cfg["model"].get("policy", "CnnPolicy"),
        env=env,
        tensorboard_log=str(tensorboard_dir),
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
        verbose=cfg["logging"].get("verbose", 1),
        device=device,
    )

    total_timesteps = cfg["experiment"]["total_timesteps"]
    callbacks = build_callbacks(cfg, eval_env, total_timesteps)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if callbacks else None,
        progress_bar=True,
    )

    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    model_path = checkpoint_dir / f"{run_name}_final"
    model.save(model_path)

    episode_returns, episode_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=cfg["experiment"].get("eval_episodes", 5),
        deterministic=True,
        return_episode_rewards=True,
    )
    mean_return = statistics.fmean(episode_returns)
    mean_length = statistics.fmean(episode_lengths)
    print(f"[train] Evaluation mean return {mean_return:.2f} over {len(episode_returns)} episodes")
    print(f"[train] Evaluation mean length {mean_length:.1f} steps")

    record_eval_video(cfg, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Atari agent with Stable-Baselines3")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ppo_breakout.yaml"),
        help="Path to a YAML configuration file",
    )
    args = parser.parse_args()

    main(args.config)
