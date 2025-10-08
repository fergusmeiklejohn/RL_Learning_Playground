# Simple Atari RL Playground — Status Snapshot (2025-10-07)

## Project Snapshot
- Focus: Atari Breakout agents exploring structured representations (RAM/pixel detectors, slot attention) and hierarchical option-based control.
- Latest run: hierarchical DQN with intrinsic options (`breakout_hier_options_intrinsic`) for 3M steps; training collapsed late and evaluations stayed near-random (reward ≈2.7).
- `experiments.md` documents every experiment chronologically; this snapshot highlights the current pause point and how to resume.

## Environment & Tooling
- Conda env: `simple-game`. Update via  
  `conda env update -n simple-game -f env/environment.yml --prune`
- Direct Python path: `/Users/fergusmeiklejohn/miniconda3/envs/simple-game/bin/python`
- Training entry point: `python -m src.simple_game.train --config <config_path>`
- Evaluation helper: `python -m src.simple_game.evaluate --config <config_path> --checkpoint <checkpoint_or_dir> [--deterministic] [--events]`

## Experiment Timeline (abridged)
- **PPO baseline** (`configs/ppo_breakout.yaml`): 2M steps, deterministic reward ≈23; establishes pipeline stability on MPS.
- **Dueling DQN + replay** (`configs/dqn_breakout.yaml`): 3M steps, higher mean reward but still unstable wins.
- **Object-centric detectors**  
  - RAM detector: deterministic ≈35 reward with strong serve reliability.  
  - Pixel detector: similar mean reward but better stochastic exploration.  
  - Hybrid RAM+pixel: combines reliable serves with tighter variance; stochastic play still trails pixel-only.
- **Slot attention encoder** (`configs/objcentric_breakout_slot.yaml`): adds reconstruction/entropy regularisers; deterministic play solid, stochastic still fragile—see `experiments.md:198-365`.
- **Hierarchical controllers**  
  - Initial no-intrinsic version stagnated near reward ≈1 (see `experiments.md:383-409`).  
  - Intrinsic options run (current) improved option bookkeeping but failed to lift scores (details below).

## Hierarchical Intrinsic Options Run Summary
- Config: `configs/hierarchical_breakout_options.yaml`, experiment name `breakout_hier_options_intrinsic`.
- Artifacts:  
  - TensorBoard: `runs/tensorboard/breakout_hier_options_intrinsic`  
  - Checkpoints: `runs/checkpoints/breakout_hier_options_intrinsic/{manager,skill_*.pt}`  
  - Eval summaries: `runs/eval_reports/breakout_hier_options_intrinsic/{deterministic,stochastic}/hierarchical_summary.json`
- Training behaviour: `train/avg_reward` peaked at 5.3 (≈2.59M steps) then decayed to 0; `train/avg_length` mirrored with late collapse. Manager ε annealed to 0.05; serve/tunnel options rarely triggered.
- Evaluations (30 games):  
  - Deterministic reward **2.77 ± 3.52**, zero-score rate **50%**, single 1200-step outlier inflates length mean.  
  - Stochastic reward **2.70 ± 3.98**, zero-score rate **53%**, option returns dominated by shaped bonuses rather than actual scoring.
- Diagnosis: negative intrinsic on `track_ball` + aggressive ε decay suppress long volleys; manager seldom exploits `serve_setup` or `tunnel_push`. Skill usage needs stronger separation and dwell-time monitoring.

## Outstanding Tasks Before Resuming
1. **Rebalance intrinsic shaping**: remove or reduce per-step penalties on `track_ball`, raise manager intrinsic scale, extend `serve_setup` horizon/warmup.
2. **Instrument option lifecycles**: log dwell time, termination causes (success vs horizon), and reward decomposition per option for TensorBoard.
3. **Adjust exploration schedule**: keep manager ε ≥0.2 until rewards improve; consider skill-specific ε plateaus.
4. **Short pilot**: rerun 300–500k-step preview after adjustments before committing to another 3M sweep.
5. **Stretch goals**: slot-attention + hierarchical hybrid, curriculum serves, or RAM-triggered options once base hierarchical system stabilises.

## Quick Commands
- Launch adjusted hierarchical run:  
  `conda run -n simple-game python -m src.simple_game.train --config configs/hierarchical_breakout_options.yaml`
- Evaluate latest checkpoints (deterministic & stochastic):  
  ```
  python -m src.simple_game.evaluate --config configs/hierarchical_breakout_options.yaml \
    --checkpoint runs/checkpoints/breakout_hier_options_intrinsic --num-games 30 --events --deterministic
  python -m src.simple_game.evaluate --config configs/hierarchical_breakout_options.yaml \
    --checkpoint runs/checkpoints/breakout_hier_options_intrinsic --num-games 30 --events
  ```
- Inspect tensorboard: `tensorboard --logdir runs/tensorboard/breakout_hier_options_intrinsic`

## Notes for Next Contributor
- Keep Gymnasium imports lazy (see `src/simple_game/train.py`) to avoid macOS/MPS segfaults.
- Use `SimpleNatureCNN` feature extractor for SB3 policies; default NatureCNN still crashes on MPS.
- ROMs expected at `~/.gymnasium/atari_roms`; AutoROM setup already run.
- When editing `env/environment.yml`, rerun the env update command to keep pip deps in sync.
- All experimental rationale and detailed metrics are appended to `experiments.md` up to line 451.
