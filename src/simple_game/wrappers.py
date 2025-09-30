"""Custom environment wrappers for instrumentation."""
from __future__ import annotations

from typing import Any, Dict, List

import gymnasium as gym


class BreakoutEventWrapper(gym.Wrapper):
    """Annotate Breakout episodes with paddle/ball event statistics.

    The wrapper tracks, per life:
    - number of FIRE (action=1) presses
    - timestep of the first FIRE
    - timestep of the first positive reward (brick hit)
    - count of positive-reward events and longest streak

    On life termination it injects a ``life_events`` dict into ``info``.
    When an episode finishes, it also emits a ``game_events`` summary that
    aggregates across lives (e.g. how many lives scored at least once).
    """

    FIRE_ACTION = 1

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._pending_game_reset = True
        self._game_life_index = 0
        self._game_lives_with_reward = 0
        self._game_total_positive_events = 0
        self._game_total_fire_presses = 0
        self._game_first_hit_steps: List[int] = []
        self._game_max_positive_streak = 0
        self._reset_life_counters()

    def _reset_life_counters(self) -> None:
        self._life_step = 0
        self._life_fire_presses = 0
        self._life_first_fire_step = -1
        self._life_first_reward_step = -1
        self._life_positive_events = 0
        self._life_reward_total = 0.0
        self._life_streak = 0
        self._life_max_streak = 0

    def _reset_game_counters(self) -> None:
        self._game_life_index = 0
        self._game_lives_with_reward = 0
        self._game_total_positive_events = 0
        self._game_total_fire_presses = 0
        self._game_first_hit_steps = []
        self._game_max_positive_streak = 0

    def reset(self, **kwargs):  # type: ignore[override]
        if self._pending_game_reset:
            self._reset_game_counters()
            self._pending_game_reset = False

        result = self.env.reset(**kwargs)
        self._reset_life_counters()
        return result

    def step(self, action):  # type: ignore[override]
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result
            terminated = truncated = done

        info = self._augment_info(info, action, float(reward), done)

        if len(result) == 5:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info

    def _augment_info(self, info: Dict[str, Any], action: Any, reward: float, done: bool) -> Dict[str, Any]:
        info = dict(info)

        action_val = self._extract_action(action)
        self._life_step += 1

        if action_val == self.FIRE_ACTION:
            self._life_fire_presses += 1
            if self._life_first_fire_step == -1:
                self._life_first_fire_step = self._life_step

        if reward > 0.0:
            self._life_positive_events += 1
            if self._life_first_reward_step == -1:
                self._life_first_reward_step = self._life_step
            self._life_streak += 1
            if self._life_streak > self._life_max_streak:
                self._life_max_streak = self._life_streak
        else:
            self._life_streak = 0

        self._life_reward_total += reward

        if done:
            life_events = {
                "life_index": self._game_life_index,
                "fire_presses": self._life_fire_presses,
                "first_fire_step": self._life_first_fire_step,
                "first_reward_step": self._life_first_reward_step,
                "positive_events": self._life_positive_events,
                "max_positive_streak": self._life_max_streak,
                "reward_total": self._life_reward_total,
                "steps": self._life_step,
            }
            info["life_events"] = life_events

            self._game_total_fire_presses += self._life_fire_presses
            self._game_total_positive_events += self._life_positive_events

            if self._life_positive_events > 0:
                self._game_lives_with_reward += 1
                self._game_first_hit_steps.append(self._life_first_reward_step)
            self._game_max_positive_streak = max(self._game_max_positive_streak, self._life_max_streak)

            self._game_life_index += 1
            self._reset_life_counters()

            if "episode" in info:
                total_lives = self._game_life_index
                positive_ratio = (
                    self._game_lives_with_reward / total_lives if total_lives else 0.0
                )
                avg_first_hit = (
                    sum(self._game_first_hit_steps) / len(self._game_first_hit_steps)
                    if self._game_first_hit_steps
                    else -1.0
                )
                info["game_events"] = {
                    "total_lives": total_lives,
                    "lives_with_reward": self._game_lives_with_reward,
                    "positive_life_ratio": positive_ratio,
                    "total_positive_events": self._game_total_positive_events,
                    "total_fire_presses": self._game_total_fire_presses,
                    "avg_first_hit_step": avg_first_hit,
                    "max_positive_streak": self._game_max_positive_streak,
                }

                self._pending_game_reset = True

        return info

    @staticmethod
    def _extract_action(action: Any) -> int:
        if isinstance(action, (list, tuple)):
            action = action[0]
        if hasattr(action, "item"):
            try:
                return int(action.item())
            except Exception:  # pragma: no cover - safety net
                pass
        try:
            return int(action)
        except Exception:  # pragma: no cover - fallback to noop
            return 0


__all__ = ["BreakoutEventWrapper"]
