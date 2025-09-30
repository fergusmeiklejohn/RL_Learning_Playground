# Evaluation Tooling Cheatsheet

We rely on a single CLI entrypoint to evaluate any SB3 checkpoint in this repo. The command wraps
Stable-Baselines3’s `evaluate_policy`, collects high-sample statistics, and optionally instruments
Breakout with extra diagnostics.

## Running the evaluator

```bash
python -m src.simple_game.evaluate \
  --config configs/ppo_breakout.yaml \
  --checkpoint runs/checkpoints/breakout_ppo_baseline_final.zip \
  --num-games 30 \
  --seeds 0 1 2 \
  --deterministic \
  --events \
  --output-dir runs/eval_reports/ppo_det
```

Key flags:

- `--num-games`: number of full episodes (per-game stats). Per-life metrics are collected
  automatically since the env resets on each life by default.
- `--seeds`: offsets added to the training config’s seed. Passing multiple values lets us report
  variance across evaluation seeds (e.g., `--seeds 0 1 2`).
- `--deterministic`: greedily selects the argmax action; omit for stochastic runs to quantify how
  exploration noise affects the learned policy.
- `--events`: wraps the env with `BreakoutEventWrapper`, logging per-life FIRE presses, first-hit
  step, positive streaks, and aggregated per-game summaries.
- `--output-dir`: dumps CSV/JSON exports. We typically mirror
  `runs/eval_reports/<experiment>/<mode>` for deterministic/stochastic runs.

## Console summary

For each seed offset we print:

- Per-life and per-game means/stdev (mirrors TensorBoard metrics).
- SB3 `evaluate_policy` results for parity with built-in evaluation callbacks.
- Failure stats: number of zero-reward lives, mean FIRE presses for those lives, and the count of
  “high-FIRE” failures (default ≥3 FIRE presses without scoring). These highlight serve issues.

## Exported files

```
per_life.csv
  seed_offset, seed, deterministic flag
  game_index, life_index, reward, length (frames)
  positive_events, first_positive_step
  fire_presses, first_fire_step, max_positive_streak
  episode/frame numbers (from Gym monitor)

per_game.csv
  cumulative reward/length per game
  lives played and lives that scored at least once
  total positive events, total FIRE presses
  average first_hit step, max positive streak, positive_life_ratio

summary.json
  echo of the CLI arguments + per-seed stats (per-life, per-game, SB3) and
  aggregate means across seeds when more than one offset is provided.
```

These exports feed directly into `notebooks/breakout_eval_dashboard.ipynb`, which visualises:

- Histogram of `first_positive_step` to detect slow starts or serve failures.
- Per-game line plots of positive events per life (brick-hit streaks).
- Tables of high-FIRE zero-reward lives for quick inspection.

## Workflow tips

1. After each training run, execute the evaluator twice—once with `--deterministic` and once without—to capture both greedy and stochastic behaviour.
2. Commit the CSV/JSON outputs if they inform documentation (they live under `runs/`, so copy
   derived plots or stats into `docs/` if needed; the raw files remain gitignored).
3. Use the console failure summary to guard against regressions (e.g., spike in zero-reward lives).
4. When experimenting with alternate configs (seeds, frame skip, exploration schedules), keep the
   `--output-dir` unique so results don’t overwrite each other.
