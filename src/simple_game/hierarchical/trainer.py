"""Training scaffolding for hierarchical controllers."""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch as th
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency for richer progress display
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm missing
    tqdm = None

from ..train import build_env, ensure_dirs
from .config import HierarchicalConfig
from .controller import HierarchicalController


@dataclass
class TrainerArtifacts:
    """Paths and handles produced during hierarchical training."""

    checkpoint_dir: str
    manager_path: Optional[str]
    skill_paths: Dict[str, str]
    training_steps: int


class ReplayBuffer:
    """Simple FIFO replay buffer storing image observations."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: th.device) -> None:
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.uint8)
        self.idx = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.actions[self.idx, 0] = action
        self.rewards[self.idx, 0] = reward
        self.dones[self.idx, 0] = 1 if done else 0
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> dict[str, th.Tensor]:
        indices = np.random.choice(self.size, batch_size, replace=False)
        obs = th.from_numpy(self.obs[indices]).float().to(self.device) / 255.0
        next_obs = th.from_numpy(self.next_obs[indices]).float().to(self.device) / 255.0
        actions = th.from_numpy(self.actions[indices]).long().to(self.device)
        rewards = th.from_numpy(self.rewards[indices]).float().to(self.device)
        dones = th.from_numpy(self.dones[indices]).float().to(self.device)
        return {
            "obs": obs,
            "next_obs": next_obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }


class DQNAgent:
    """Lightweight DQN agent driving either manager or skill policies."""

    def __init__(
        self,
        policy: th.nn.Module,
        action_dim: int,
        obs_shape: tuple[int, ...],
        *,
        device: th.device,
        buffer_size: int,
        batch_size: int,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        gradient_updates_per_step: int,
    ) -> None:
        self.policy = policy.to(device)
        self.target = copy.deepcopy(policy).to(device)
        self.target.eval()
        self.action_dim = action_dim
        self.replay = ReplayBuffer(buffer_size, obs_shape, device)
        self.optimizer = th.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.gradient_updates_per_step = gradient_updates_per_step
        self.batch_size = batch_size
        self.device = device
        self.train_steps = 0
        self.total_updates = 0

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        obs_tensor = th.from_numpy(obs).float().unsqueeze(0).to(self.device) / 255.0
        with th.no_grad():
            q_values = self.policy(obs_tensor)
        return int(q_values.argmax(dim=1).item())

    def store(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.replay.add(obs, action, reward, next_obs, done)

    def update(self) -> Optional[float]:
        if not self.replay.can_sample(self.batch_size):
            return None
        losses = []
        for _ in range(self.gradient_updates_per_step):
            batch = self.replay.sample(self.batch_size)
            obs = batch["obs"]
            next_obs = batch["next_obs"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            dones = batch["dones"]

            q_values = self.policy(obs).gather(1, actions)
            with th.no_grad():
                next_q = self.target(next_obs).max(dim=1, keepdim=True).values
                target = rewards + (1.0 - dones) * self.gamma * next_q
            loss = F.smooth_l1_loss(q_values, target)
            self.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.optimizer.step()
            losses.append(float(loss.item()))
            self.total_updates += 1
            if self.total_updates % self.target_update_interval == 0:
                self.target.load_state_dict(self.policy.state_dict())
        return float(np.mean(losses)) if losses else None

    def to_eval(self) -> None:
        self.policy.eval()

    def to_train(self) -> None:
        self.policy.train()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        th.save({"policy": self.policy.state_dict(), "target": self.target.state_dict()}, path)


class HierarchicalTrainer:
    """Co-trains manager and skill policies using simple DQN updates."""

    def __init__(self, base_cfg: dict, cfg: HierarchicalConfig) -> None:
        self.base_cfg = base_cfg
        self.cfg = cfg
        self.device = th.device(cfg.device or ("mps" if th.backends.mps.is_available() else "cpu"))
        self.controller: Optional[HierarchicalController] = None

    def build_controller(self, obs_space, action_space) -> HierarchicalController:
        controller = HierarchicalController(obs_space, action_space, self.cfg, str(self.device))
        self.controller = controller
        return controller

    def _epsilon(self, start: float, end: float, decay_steps: int, step: int) -> float:
        if decay_steps <= 0:
            return end
        fraction = min(1.0, step / decay_steps)
        return float(start + fraction * (end - start))

    def train(self) -> TrainerArtifacts:
        ensure_dirs(self.base_cfg)
        env = build_env(self.base_cfg, mode="train", n_envs=1, event_wrapper=True)
        observation_space = env.observation_space
        action_space = env.action_space
        controller = self.build_controller(observation_space, action_space)

        skill_agents: Dict[str, DQNAgent] = {}
        for skill_name, skill_policy in controller.skills.items():
            skill_agents[skill_name] = DQNAgent(
                policy=skill_policy,
                action_dim=action_space.n,
                obs_shape=observation_space.shape,
                device=self.device,
                buffer_size=self.cfg.buffer_size,
                batch_size=self.cfg.batch_size,
                learning_rate=self.cfg.learning_rate,
                gamma=self.cfg.gamma,
                target_update_interval=self.cfg.target_update_interval,
                gradient_updates_per_step=self.cfg.gradient_updates_per_step,
            )

        manager_agent = DQNAgent(
            policy=controller.manager,
            action_dim=len(controller.skill_names()),
            obs_shape=observation_space.shape,
            device=self.device,
            buffer_size=self.cfg.buffer_size,
            batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.manager_learning_rate,
            gamma=self.cfg.gamma,
            target_update_interval=self.cfg.target_update_interval,
            gradient_updates_per_step=self.cfg.gradient_updates_per_step,
        )

        obs_vec = env.reset()
        obs = np.array(obs_vec[0])
        current_skill: Optional[str] = None
        current_skill_index: Optional[int] = None
        option_reward = 0.0
        option_steps = 0
        option_start_obs = obs.copy()
        episode_reward = 0.0
        episode_length = 0
        episode_rewards = []
        episode_lengths = []
        manager_losses = []
        skill_losses = []

        total_steps = self.cfg.total_timesteps
        log_interval = self.base_cfg.get("logging", {}).get("log_interval", 1000)
        next_eval = self.cfg.eval_interval
        next_checkpoint = self.cfg.checkpoint_interval

        progress = None
        if self.base_cfg.get("experiment", {}).get("progress_bar", True) and tqdm is not None:
            progress = tqdm(total=total_steps, desc="Hierarchical training", dynamic_ncols=True)

        for step in range(1, total_steps + 1):
            if current_skill is None:
                epsilon = self._epsilon(
                    self.cfg.manager_epsilon_start,
                    self.cfg.manager_epsilon_end,
                    self.cfg.manager_epsilon_decay_steps,
                    step,
                )
                current_skill_index = manager_agent.select_action(obs, epsilon)
                current_skill = controller.skill_names()[current_skill_index]
                option_reward = 0.0
                option_steps = 0
                option_start_obs = obs.copy()

            skill_cfg = controller.skill_config(current_skill)
            skill_agent = skill_agents[current_skill]
            epsilon_skill = self._epsilon(
                self.cfg.epsilon_start,
                self.cfg.epsilon_end,
                self.cfg.epsilon_decay_steps,
                step,
            )
            action = skill_agent.select_action(obs, epsilon_skill)
            step_action = np.array([action], dtype=np.int64)
            next_obs_vec, rewards_vec, dones_vec, infos = env.step(step_action)
            reward = float(rewards_vec[0])
            done = bool(dones_vec[0])
            next_obs = np.array(next_obs_vec[0])

            skill_agent.store(obs.copy(), action, reward, next_obs.copy(), done)
            if step > self.cfg.skill_warmup:
                loss = skill_agent.update()
                if loss is not None:
                    skill_losses.append(loss)

            option_reward += reward
            option_steps += 1

            episode_reward += reward
            episode_length += 1

            obs = next_obs

            terminated = done or option_steps >= skill_cfg.horizon
            if skill_cfg.termination_on_success and option_reward > 0:
                terminated = True

            if terminated and current_skill is not None and current_skill_index is not None:
                bonus = skill_cfg.success_reward if option_reward > 0 else skill_cfg.failure_penalty
                manager_reward = option_reward + bonus
                manager_agent.store(option_start_obs.copy(), current_skill_index, manager_reward, obs.copy(), done)
                if step > self.cfg.high_level_warmup:
                    loss = manager_agent.update()
                    if loss is not None:
                        manager_losses.append(loss)
                current_skill = None
                current_skill_index = None

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                obs_vec = env.reset()
                obs = np.array(obs_vec[0])
                episode_reward = 0.0
                episode_length = 0
                current_skill = None
                current_skill_index = None
                option_reward = 0.0
                option_steps = 0
                option_start_obs = obs.copy()

            if progress is not None:
                progress.update(1)

            if log_interval and step % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else float("nan")
                avg_len = np.mean(episode_lengths[-10:]) if episode_lengths else float("nan")
                avg_skill_loss = np.mean(skill_losses[-log_interval:]) if skill_losses else float("nan")
                avg_manager_loss = np.mean(manager_losses[-log_interval:]) if manager_losses else float("nan")
                if progress is not None:
                    progress.set_postfix(
                        {
                            "reward": "%.2f" % avg_reward,
                            "len": "%.1f" % avg_len,
                            "mgr_eps": "%.3f"
                            % self._epsilon(
                                self.cfg.manager_epsilon_start,
                                self.cfg.manager_epsilon_end,
                                self.cfg.manager_epsilon_decay_steps,
                                step,
                            ),
                            "sk_eps": "%.3f"
                            % self._epsilon(
                                self.cfg.epsilon_start,
                                self.cfg.epsilon_end,
                                self.cfg.epsilon_decay_steps,
                                step,
                            ),
                        }
                    )
                else:
                    print(
                        f"[hier] step={step} avg_reward={avg_reward:.2f} avg_len={avg_len:.1f}"
                        f" manager_eps={self._epsilon(self.cfg.manager_epsilon_start, self.cfg.manager_epsilon_end, self.cfg.manager_epsilon_decay_steps, step):.3f}"
                        f" skill_eps={self._epsilon(self.cfg.epsilon_start, self.cfg.epsilon_end, self.cfg.epsilon_decay_steps, step):.3f}"
                        f" skill_loss={avg_skill_loss:.4f} manager_loss={avg_manager_loss:.4f}"
                    )

            if self.cfg.eval_interval and step >= next_eval:
                self._evaluate(env, manager_agent, skill_agents)
                next_eval += self.cfg.eval_interval

            if self.cfg.checkpoint_interval and step >= next_checkpoint:
                self._save_checkpoints(manager_agent, skill_agents)
                next_checkpoint += self.cfg.checkpoint_interval

        checkpoint_dir, manager_path, skill_paths = self._save_checkpoints(manager_agent, skill_agents)
        if progress is not None:
            progress.close()
        env.close()
        return TrainerArtifacts(
            checkpoint_dir=str(checkpoint_dir),
            manager_path=str(manager_path) if manager_path else None,
            skill_paths={k: str(v) for k, v in skill_paths.items()},
            training_steps=total_steps,
        )

    def _evaluate(self, env, manager_agent: DQNAgent, skill_agents: Dict[str, DQNAgent]) -> None:
        eval_env = build_env(self.base_cfg, mode="eval", n_envs=1, event_wrapper=True)
        obs_vec = eval_env.reset()
        obs = np.array(obs_vec[0])
        total_rewards = []
        total_lengths = []
        skill_names = self.controller.skill_names() if self.controller else []
        for _ in range(self.cfg.eval_games):
            done = False
            episode_reward = 0.0
            episode_len = 0
            current_skill: Optional[str] = None
            current_index: Optional[int] = None
            option_reward = 0.0
            option_steps = 0
            while not done:
                if current_skill is None:
                    action_idx = manager_agent.select_action(obs, epsilon=0.0)
                    current_skill = skill_names[action_idx]
                    current_index = action_idx
                    option_reward = 0.0
                    option_steps = 0
                skill_cfg = self.controller.skill_config(current_skill) if self.controller else None
                agent = skill_agents[current_skill]
                action = agent.select_action(obs, epsilon=0.0)
                next_obs_vec, rewards_vec, dones_vec, _ = eval_env.step(np.array([action], dtype=np.int64))
                reward = float(rewards_vec[0])
                done = bool(dones_vec[0])
                obs = np.array(next_obs_vec[0])
                episode_reward += reward
                episode_len += 1
                option_reward += reward
                option_steps += 1
                if skill_cfg and (done or option_steps >= skill_cfg.horizon or (skill_cfg.termination_on_success and option_reward > 0)):
                    current_skill = None
                    current_index = None
                    option_reward = 0.0
                    option_steps = 0
                if done:
                    total_rewards.append(episode_reward)
                    total_lengths.append(episode_len)
                    obs_vec = eval_env.reset()
                    obs = np.array(obs_vec[0])
        if total_rewards:
            print(
                f"[hier-eval] reward_mean={np.mean(total_rewards):.2f}"
                f" ±{np.std(total_rewards):.2f} length_mean={np.mean(total_lengths):.1f}"
            )
        eval_env.close()

    def _save_checkpoints(self, manager_agent: DQNAgent, skill_agents: Dict[str, DQNAgent]) -> tuple[Path, Optional[Path], Dict[str, Path]]:
        checkpoint_dir = Path(self.base_cfg["logging"]["checkpoint_dir"]) / self.base_cfg["experiment"]["name"]
        manager_path: Optional[Path] = None
        skill_paths: Dict[str, Path] = {}
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manager_path = checkpoint_dir / "manager.pt"
        manager_agent.save(manager_path)
        for name, agent in skill_agents.items():
            path = checkpoint_dir / f"skill_{name}.pt"
            agent.save(path)
            skill_paths[name] = path
        return checkpoint_dir, manager_path, skill_paths

    def evaluate_from_checkpoints(
        self,
        checkpoint_dir: Path,
        *,
        num_games: int = 30,
        deterministic: bool = True,
        event_wrapper: bool = True,
    ) -> dict:
        """Run greedy (or epsilon-soft) rollouts from saved manager/skills checkpoints."""

        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory '{checkpoint_dir}' not found")

        env = build_env(self.base_cfg, mode="eval", n_envs=1, event_wrapper=event_wrapper)
        obs_space = env.observation_space
        action_space = env.action_space

        controller = HierarchicalController(obs_space, action_space, self.cfg, str(self.device))
        manager_path = checkpoint_dir / "manager.pt"
        if not manager_path.exists():
            raise FileNotFoundError(f"Manager checkpoint '{manager_path}' not found")

        manager_state = th.load(manager_path, map_location=self.device)
        controller.manager.load_state_dict(manager_state["policy"])

        # Build evaluation-only agents (no learning / gradient updates)
        manager_agent = DQNAgent(
            controller.manager,
            len(controller.skill_names()),
            obs_space.shape,
            device=self.device,
            buffer_size=1,
            batch_size=1,
            learning_rate=self.cfg.manager_learning_rate,
            gamma=self.cfg.gamma,
            target_update_interval=self.cfg.target_update_interval,
            gradient_updates_per_step=0,
        )
        manager_agent.policy.load_state_dict(manager_state["policy"])
        manager_agent.target.load_state_dict(manager_state["target"])
        manager_agent.to_eval()

        skill_agents: Dict[str, DQNAgent] = {}
        for name, skill in controller.skills.items():
            skill_path = checkpoint_dir / f"skill_{name}.pt"
            if not skill_path.exists():
                raise FileNotFoundError(f"Skill checkpoint '{skill_path}' not found")
            state = th.load(skill_path, map_location=self.device)
            agent = DQNAgent(
                skill,
                action_space.n,
                obs_space.shape,
                device=self.device,
                buffer_size=1,
                batch_size=1,
                learning_rate=self.cfg.learning_rate,
                gamma=self.cfg.gamma,
                target_update_interval=self.cfg.target_update_interval,
                gradient_updates_per_step=0,
            )
            agent.policy.load_state_dict(state["policy"])
            agent.target.load_state_dict(state["target"])
            agent.to_eval()
            skill_agents[name] = agent

        manager_eps = 0.0 if deterministic else self.cfg.manager_epsilon_end
        skill_eps = 0.0 if deterministic else self.cfg.epsilon_end

        skill_names = controller.skill_names()
        skill_usage = {name: 0 for name in skill_names}
        skill_return_summaries = {name: [] for name in skill_names}

        episode_rewards: list[float] = []
        episode_lengths: list[int] = []

        for _ in range(num_games):
            obs_vec = env.reset()
            obs = np.array(obs_vec[0])
            done = False
            ep_reward = 0.0
            ep_length = 0
            current_skill: Optional[str] = None
            current_agent: Optional[DQNAgent] = None
            option_reward = 0.0
            option_steps = 0

            while not done:
                if current_skill is None:
                    idx = manager_agent.select_action(obs, epsilon=manager_eps)
                    current_skill = skill_names[idx]
                    current_agent = skill_agents[current_skill]
                    skill_usage[current_skill] += 1
                    option_reward = 0.0
                    option_steps = 0

                skill_cfg = controller.skill_config(current_skill)
                action = current_agent.select_action(obs, epsilon=skill_eps)
                next_obs_vec, rewards_vec, dones_vec, _ = env.step(np.array([action], dtype=np.int64))
                reward = float(rewards_vec[0])
                done = bool(dones_vec[0])
                obs = np.array(next_obs_vec[0])

                ep_reward += reward
                ep_length += 1
                option_reward += reward
                option_steps += 1

                terminate = done or option_steps >= skill_cfg.horizon
                if skill_cfg.termination_on_success and option_reward > 0:
                    terminate = True

                if terminate:
                    skill_return_summaries[current_skill].append(option_reward)
                    current_skill = None
                    current_agent = None

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

        env.close()

        rewards_np = np.asarray(episode_rewards, dtype=np.float32)
        lengths_np = np.asarray(episode_lengths, dtype=np.float32)
        zero_ratio = float((rewards_np == 0.0).mean()) if rewards_np.size else float("nan")

        skill_option_means = {
            name: (float(np.mean(vals)) if vals else float("nan"))
            for name, vals in skill_return_summaries.items()
        }

        return {
            "games": num_games,
            "reward_mean": float(rewards_np.mean()) if rewards_np.size else float("nan"),
            "reward_std": float(rewards_np.std()) if rewards_np.size else float("nan"),
            "length_mean": float(lengths_np.mean()) if lengths_np.size else float("nan"),
            "length_std": float(lengths_np.std()) if lengths_np.size else float("nan"),
            "zero_reward_fraction": zero_ratio,
            "skill_usage": skill_usage,
            "skill_option_mean": skill_option_means,
        }
