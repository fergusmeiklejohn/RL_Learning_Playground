"""Evaluation utilities for Atari agents."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from .train import build_env, load_config, select_device


@dataclass
class Stats:
    """Simple summary container for a sequence of numeric values."""

    count: int
    mean: float
    stdev: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "stdev": self.stdev,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass
class LifeRecord:
    seed_offset: int
    seed: int
    deterministic: bool
    game_index: int
    life_index: int
    reward: float
    length: int
    positive_events: int
    first_positive_step: int
    fire_presses: int
    first_fire_step: int
    max_positive_streak: int
    episode_frame_number: Optional[int]
    frame_number: Optional[int]


@dataclass
class GameRecord:
    seed_offset: int
    seed: int
    deterministic: bool
    game_index: int
    reward: float
    length: int
    lives: int
    lives_with_reward: int
    total_positive_events: int
    total_fire_presses: int
    avg_first_positive_step: float
    max_positive_streak: int
    positive_life_ratio: float


@dataclass
class EvaluationResult:
    seed_offset: int
    deterministic: bool
    seed: int
    per_life_rewards: Stats
    per_life_lengths: Stats
    per_game_rewards: Stats
    per_game_lengths: Stats
    sb3_rewards: Stats
    sb3_lengths: Stats
    per_life_records: List[LifeRecord]
    per_game_records: List[GameRecord]
    failure_stats: Optional[dict] = None


def summarize(values: Sequence[float]) -> Stats:
    """Compute count/mean/stdev/min/max without importing NumPy."""

    data = list(float(v) for v in values)
    if not data:
        return Stats(count=0, mean=float("nan"), stdev=float("nan"), minimum=float("nan"), maximum=float("nan"))

    mean = statistics.fmean(data)
    stdev = statistics.stdev(data) if len(data) > 1 else 0.0
    return Stats(count=len(data), mean=mean, stdev=stdev, minimum=min(data), maximum=max(data))


def evaluate_checkpoint(
    cfg: dict,
    checkpoint_path: Path,
    *,
    deterministic: bool,
    num_games: int,
    seed_offset: int,
    seed: int,
    device: str,
    event_wrapper: bool,
) -> EvaluationResult:
    """Run evaluation collecting both per-life and per-game statistics."""

    env = build_env(
        cfg,
        mode="eval",
        n_envs=1,
        seed_offset=seed_offset,
        monitor_dir=None,
        event_wrapper=event_wrapper,
    )
    model = PPO.load(checkpoint_path, device=device)

    per_life_rewards: List[float] = []
    per_life_lengths: List[int] = []
    per_game_rewards: List[float] = []
    per_game_lengths: List[int] = []
    per_life_records: List[LifeRecord] = []
    per_game_records: List[GameRecord] = []

    obs = env.reset()

    life_reward = 0.0
    life_length = 0
    life_positive_events = 0
    life_fire_presses = 0
    life_first_positive_step: Optional[int] = None
    life_first_fire_step: Optional[int] = None
    life_current_streak = 0
    life_max_positive_streak = 0

    life_index_in_game = 0
    game_life_count = 0
    game_positive_life_count = 0
    game_total_positive_events = 0
    game_total_fire_presses = 0
    game_first_hit_steps: List[int] = []
    game_max_positive_streak = 0

    def extract_action_val(action_obj) -> int:
        if isinstance(action_obj, (list, tuple)):
            action_obj = action_obj[0]
        if hasattr(action_obj, "item"):
            try:
                return int(action_obj.item())
            except Exception:
                pass
        try:
            return int(action_obj)
        except Exception:
            return 0

    games_collected = 0
    while games_collected < num_games:
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(actions)

        reward = float(rewards[0])
        life_reward += reward
        life_length += 1

        action_val = extract_action_val(actions[0] if isinstance(actions, (list, tuple)) else actions)
        if action_val == 1:
            life_fire_presses += 1
            if life_first_fire_step is None:
                life_first_fire_step = life_length

        if reward > 0.0:
            life_positive_events += 1
            life_current_streak += 1
            if life_first_positive_step is None:
                life_first_positive_step = life_length
            if life_current_streak > life_max_positive_streak:
                life_max_positive_streak = life_current_streak
        else:
            life_current_streak = 0

        if dones[0]:
            life_info = infos[0].get("life_events")
            if life_info:
                life_fire_presses = int(life_info.get("fire_presses", life_fire_presses))
                life_first_fire_step = life_info.get("first_fire_step", life_first_fire_step)
                life_positive_events = int(life_info.get("positive_events", life_positive_events))
                life_first_positive_step = life_info.get("first_reward_step", life_first_positive_step)
                life_max_positive_streak = int(
                    life_info.get("max_positive_streak", life_max_positive_streak)
                )

            first_positive_val = (
                int(life_first_positive_step)
                if life_first_positive_step is not None
                else -1
            )
            first_fire_val = (
                int(life_first_fire_step)
                if life_first_fire_step is not None
                else -1
            )

            per_life_rewards.append(life_reward)
            per_life_lengths.append(life_length)
            per_life_records.append(
                LifeRecord(
                    seed_offset=seed_offset,
                    seed=seed,
                    deterministic=deterministic,
                    game_index=games_collected,
                    life_index=life_index_in_game,
                    reward=life_reward,
                    length=life_length,
                    positive_events=life_positive_events,
                    first_positive_step=first_positive_val,
                    fire_presses=life_fire_presses,
                    first_fire_step=first_fire_val,
                    max_positive_streak=life_max_positive_streak,
                    episode_frame_number=infos[0].get("episode_frame_number"),
                    frame_number=infos[0].get("frame_number"),
                )
            )

            game_life_count += 1
            if life_positive_events > 0:
                game_positive_life_count += 1
                if first_positive_val != -1:
                    game_first_hit_steps.append(first_positive_val)
            game_total_positive_events += life_positive_events
            game_total_fire_presses += life_fire_presses
            game_max_positive_streak = max(game_max_positive_streak, life_max_positive_streak)

            life_reward = 0.0
            life_length = 0
            life_positive_events = 0
            life_fire_presses = 0
            life_first_positive_step = None
            life_first_fire_step = None
            life_current_streak = 0
            life_max_positive_streak = 0
            life_index_in_game += 1

        episode_info = infos[0].get("episode")
        if episode_info is not None:
            lives = game_life_count
            lives_with_reward = game_positive_life_count
            avg_first_hit = (
                sum(game_first_hit_steps) / len(game_first_hit_steps)
                if game_first_hit_steps
                else -1.0
            )
            positive_ratio = lives_with_reward / lives if lives else 0.0

            game_events = infos[0].get("game_events")
            if game_events:
                lives = int(game_events.get("total_lives", lives))
                lives_with_reward = int(game_events.get("lives_with_reward", lives_with_reward))
                positive_ratio = float(
                    game_events.get("positive_life_ratio", positive_ratio)
                )
                avg_first_hit = float(
                    game_events.get("avg_first_hit_step", avg_first_hit)
                )
                game_total_positive_events = int(
                    game_events.get("total_positive_events", game_total_positive_events)
                )
                game_total_fire_presses = int(
                    game_events.get("total_fire_presses", game_total_fire_presses)
                )
                game_max_positive_streak = int(
                    game_events.get("max_positive_streak", game_max_positive_streak)
                )
                if lives:
                    life_index_in_game = lives

            per_game_rewards.append(float(episode_info["r"]))
            per_game_lengths.append(int(episode_info["l"]))
            per_game_records.append(
                GameRecord(
                    seed_offset=seed_offset,
                    seed=seed,
                    deterministic=deterministic,
                    game_index=games_collected,
                    reward=float(episode_info["r"]),
                    length=int(episode_info["l"]),
                    lives=lives,
                    lives_with_reward=lives_with_reward,
                    total_positive_events=game_total_positive_events,
                    total_fire_presses=game_total_fire_presses,
                    avg_first_positive_step=avg_first_hit,
                    max_positive_streak=game_max_positive_streak,
                    positive_life_ratio=positive_ratio,
                )
            )
            games_collected += 1
            life_index_in_game = 0
            game_life_count = 0
            game_positive_life_count = 0
            game_total_positive_events = 0
            game_total_fire_presses = 0
            game_first_hit_steps = []
            game_max_positive_streak = 0

    sb3_returns, sb3_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=num_games,
        deterministic=deterministic,
        return_episode_rewards=True,
    )

    env.close()

    failure_stats = analyze_failures(per_life_records)

    return EvaluationResult(
        seed_offset=seed_offset,
        deterministic=deterministic,
        seed=seed,
        per_life_rewards=summarize(per_life_rewards),
        per_life_lengths=summarize(per_life_lengths),
        per_game_rewards=summarize(per_game_rewards),
        per_game_lengths=summarize(per_game_lengths),
        sb3_rewards=summarize(sb3_returns),
        sb3_lengths=summarize(sb3_lengths),
        per_life_records=per_life_records,
        per_game_records=per_game_records,
        failure_stats=failure_stats,
    )


def aggregate(results: Iterable[EvaluationResult], attr: str) -> Stats:
    """Aggregate the requested attribute across seeds (e.g., per_game_rewards.mean)."""

    values: List[float] = []
    for result in results:
        stats: Stats = getattr(result, attr)
        if stats.count == 0:
            continue
        values.append(stats.mean)
    return summarize(values)


def format_stats(label: str, stats: Stats) -> str:
    if stats.count == 0:
        return f"{label}: n=0"
    return (
        f"{label}: n={stats.count}, mean={stats.mean:.2f}, std={stats.stdev:.2f},"
        f" min={stats.minimum:.2f}, max={stats.maximum:.2f}"
    )


def analyze_failures(per_life_records: List[LifeRecord]) -> dict:
    zero_reward = [rec for rec in per_life_records if rec.positive_events == 0]
    high_fire_zero = sorted(
        zero_reward,
        key=lambda rec: (rec.fire_presses, rec.length),
        reverse=True,
    )
    threshold = 3
    high_fire_threshold = [rec for rec in zero_reward if rec.fire_presses >= threshold]

    def mean(values: List[float]) -> float:
        if not values:
            return float("nan")
        return sum(values) / len(values)

    return {
        "zero_reward_lives": len(zero_reward),
        "zero_reward_mean_fire": mean([rec.fire_presses for rec in zero_reward]),
        "zero_reward_mean_length": mean([rec.length for rec in zero_reward]),
        "high_fire_zero_count": len(high_fire_threshold),
        "high_fire_zero_examples": [
            {
                "seed_offset": rec.seed_offset,
                "seed": rec.seed,
                "game_index": rec.game_index,
                "life_index": rec.life_index,
                "fire_presses": rec.fire_presses,
                "length": rec.length,
            }
            for rec in high_fire_zero[:5]
        ],
        "high_fire_threshold": threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config used for training.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the .zip checkpoint produced by Stable-Baselines3.",
    )
    parser.add_argument("--num-games", type=int, default=30, help="Number of full games to evaluate.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Additional eval seed offsets (each adds to the config seed).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions during evaluation (default: stochastic).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device specification (auto/mps/cpu/cuda).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory to store CSV/JSON exports (created if missing).",
    )
    parser.add_argument(
        "--events",
        action="store_true",
        help="Enable Breakout event wrapper for detailed paddle/ball diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    base_seed_offset = 10_000  # match train.build_env default for eval mode
    base_seed = cfg.get("experiment", {}).get("seed", 0)

    device = select_device(args.device)

    results: List[EvaluationResult] = []
    for offset in args.seeds:
        seed_offset = base_seed_offset + offset
        actual_seed = base_seed + seed_offset
        result = evaluate_checkpoint(
            cfg,
            args.checkpoint,
            deterministic=args.deterministic,
            num_games=args.num_games,
            seed_offset=seed_offset,
            seed=actual_seed,
            device=device,
            event_wrapper=args.events,
        )
        results.append(result)

        print(
            f"Seed offset {seed_offset} (seed {actual_seed},"
            f" {'det' if args.deterministic else 'stoch'}):"
        )
        print("  " + format_stats("Per-life reward", result.per_life_rewards))
        print("  " + format_stats("Per-life length", result.per_life_lengths))
        print("  " + format_stats("Per-game reward", result.per_game_rewards))
        print("  " + format_stats("Per-game length", result.per_game_lengths))
        print("  " + format_stats("SB3 reward", result.sb3_rewards))
        print("  " + format_stats("SB3 length", result.sb3_lengths))
        if result.failure_stats:
            zero_reward = result.failure_stats["zero_reward_lives"]
            mean_fire = result.failure_stats["zero_reward_mean_fire"]
            high_fire_count = result.failure_stats["high_fire_zero_count"]
            threshold = result.failure_stats["high_fire_threshold"]
            print(
                f"  Zero-reward lives: {zero_reward}"
                f" (mean FIRE {mean_fire:.1f});"
                f" high-FIRE (≥{threshold}) zero-reward lives: {high_fire_count}"
            )
            top_examples = result.failure_stats.get("high_fire_zero_examples", [])
            if top_examples:
                exemplar = top_examples[0]
                print(
                    "    Max FIRE with zero reward: {fire_presses}"
                    " (game {game_index}, life {life_index})".format(**exemplar)
                )
        print()

    if len(results) > 1:
        per_game_mean = aggregate(results, "per_game_rewards")
        print("Aggregate per-game reward across seeds:")
        print("  " + format_stats("Mean of means", per_game_mean))

    if args.output_dir is not None:
        write_exports(args.output_dir, args, cfg, results)


def write_exports(output_dir: Path, args: argparse.Namespace, cfg: dict, results: List[EvaluationResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    life_fieldnames = [
        "seed_offset",
        "seed",
        "deterministic",
        "game_index",
        "life_index",
        "reward",
        "length",
        "positive_events",
        "first_positive_step",
        "fire_presses",
        "first_fire_step",
        "max_positive_streak",
        "episode_frame_number",
        "frame_number",
    ]
    game_fieldnames = [
        "seed_offset",
        "seed",
        "deterministic",
        "game_index",
        "reward",
        "length",
        "lives",
        "lives_with_reward",
        "total_positive_events",
        "total_fire_presses",
        "avg_first_positive_step",
        "max_positive_streak",
        "positive_life_ratio",
    ]

    life_path = output_dir / "per_life.csv"
    game_path = output_dir / "per_game.csv"
    summary_path = output_dir / "summary.json"

    with life_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=life_fieldnames)
        writer.writeheader()
        for result in results:
            for record in result.per_life_records:
                row = asdict(record)
                row["deterministic"] = int(row["deterministic"])
                writer.writerow(row)

    with game_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=game_fieldnames)
        writer.writeheader()
        for result in results:
            for record in result.per_game_records:
                row = asdict(record)
                row["deterministic"] = int(row["deterministic"])
                writer.writerow(row)

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "deterministic": args.deterministic,
        "num_games": args.num_games,
        "seeds": [int(s) for s in args.seeds],
        "base_seed": cfg.get("experiment", {}).get("seed", 0),
        "events": args.events,
        "results": [],
    }
    for result in results:
        summary["results"].append(
            {
                "seed_offset": result.seed_offset,
                "seed": result.seed,
                "deterministic": result.deterministic,
                "per_life_rewards": result.per_life_rewards.to_dict(),
                "per_life_lengths": result.per_life_lengths.to_dict(),
                "per_game_rewards": result.per_game_rewards.to_dict(),
                "per_game_lengths": result.per_game_lengths.to_dict(),
                "sb3_rewards": result.sb3_rewards.to_dict(),
                "sb3_lengths": result.sb3_lengths.to_dict(),
                "failure_stats": result.failure_stats,
            }
        )

    if len(results) > 1:
        summary["aggregate_per_game_reward"] = aggregate(results, "per_game_rewards").to_dict()

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
