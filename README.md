# Simple Atari RL Playground

This project aims to learn the basics of RL by training different reinforcement learning agents on classic Atari games using [Gymnasium](https://gymnasium.farama.org/) environments and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) on an Apple Silicon (M3 Max) Mac running macOS. The goal is to keep the workflow transparent by documenting decisions, trade-offs, and alternatives.

## Project Status & Key Learnings
- Built an Apple Silicon-friendly RL stack (custom `SimpleNatureCNN`, lazy Gymnasium imports) plus an evaluation CLI that exports per-life/per-game/Q-gap stats so every experiment runs with reproducible configs and diagnostics.
- PPO baseline (2M steps) settles near 22 reward per game (per-life ≈3.7), highlighting persistent serve-drop failures and teaching us to trust Monitor game totals rather than per-life resets when reading SB3 logs.
- Prioritized dueling Double-DQN (3M steps) reaches 33.8 ± 3.2 deterministic and 35.7 ± 1.5 stochastic reward with 4–6× fewer zero-reward lives than PPO, showing replay + dueling heads accelerate learning under sparse Breakout rewards.
- RAM-tap detector DQN pushes greedy reward to 35.46 ± 0.53, stretches games to ~1,061 frames, and cuts zero-reward lives to 9.8%, albeit with a stochastic drop to 33.24 as exploration samples tie-valued actions.
- Hybrid RAM+pixel features halve zero-reward lives again (5.8%) without widening Q-gaps (~0.34), while slot-attention + auxiliary losses widen gaps to 0.40 and deliver 31.2 reward but still trail RAM baselines under exploration.
- Hierarchical options agent trains end-to-end yet plateaus around 2.7 reward with >50% zero-reward games; upcoming work rebalances intrinsic shaping, logs option dwell times, and slows epsilon decay before the next sweep.

## High-Level Plan

1. **Environment provisioning**  
   - Manage dependencies with Conda for reproducibility and easy GPU/CPU switching.  
   - Use `pip` inside the Conda env for packages without native Conda builds (e.g., PyTorch for MPS).
2. **ROM licensing & assets**  
   - Install Atari ROMs using `AutoROM.accept-rom-license`.  
   - Document how licenses are handled and where ROMs are stored locally.
3. **Experiment scaffolding**  
   - Maintain configs (`configs/`) for hyperparameters and environment choices.  
   - Write training scripts in `src/simple_game/` with logging, checkpointing, and TensorBoard support.  
   - Provide CLI entry points in `scripts/` for setup, training, and evaluation.
4. **Training loop**  
   - Start with PPO (well-supported by SB3 for Atari).  
   - Enable MPS acceleration when available; fall back to CPU automatically.  
   - Log metrics, videos, and hyperparameters for reproducibility.
5. **Evaluation & iteration**  
   - Capture policy rollouts for qualitative inspection.  
   - Track experiment metadata in `runs/` and standardize experiment naming.  
   - Use the shared evaluation CLI to summarise ≥30-episode metrics (per-life and per-game).  
   - Iterate on hyperparameters or algorithms (e.g., dueling DQN) once baseline PPO is stable.

## Directory Layout

```
.
├── configs/            # YAML configs for environments, algorithms, logging
├── docs/               # Additional notes, design decisions, reports
├── notebooks/          # Exploratory analyses and debugging notebooks
├── runs/               # TensorBoard logs, checkpoints, videos (gitignored)
├── scripts/            # CLI helpers (env setup, training entrypoints)
└── src/simple_game/    # Python package with training modules
```

## Getting Started

1. **Create or update the Conda env**
   ```bash
   conda env create -f env/environment.yml          # first time
   conda env update -n simple-game -f env/environment.yml --prune  # later updates
   conda activate simple-game
   ```
2. **Install Atari ROMs**
   ```bash
   python -m AutoROM.accept-rom-license --install-dir ~/.gymnasium/atari_roms
   ```
3. **Run the baseline PPO training**
   ```bash
   python -m src.simple_game.train --config configs/ppo_breakout.yaml
   ```
4. **(Optional) Launch the dueling Double-DQN baseline**
   ```bash
   python -m src.simple_game.train --config configs/dqn_breakout.yaml
   ```

See below for details on each component.

---

## Why Conda + pip?

- `conda` simplifies management of system packages (SDL2, libjpeg) and isolates Python versions.
- Apple Silicon support for PyTorch with MPS is delivered through pip wheels; Conda-forge builds support `metal` only for select packages. Installing PyTorch via `pip` within a Conda env gives access to the latest MPS optimizations without waiting for Conda builds.
- Gymnasium extras (`gymnasium[atari,accept-rom-license]`) are pip-only; combining `conda` base packages with pip extras keeps the environment manageable.

### Alternatives considered

- **Docker**: reproducible but introduces virtualization overhead on macOS and complicates access to Metal/MPS. We prefer native execution for better performance.
- **venv + pip**: lighter weight but less control over non-Python dependencies needed for Atari emulation.

## Atari ROM Handling

- `AutoROM` downloads Atari ROMs from the legal distribution provided by the ROM licensing terms.  
- ROMs are stored outside the repo (`~/.gymnasium/atari_roms`) to avoid license issues and keep the repo clean.
- We track the AutoROM version in `env/environment.yml` to ensure deterministic ROM setups.

## Training Strategy

- Start with **PPO** (`configs/ppo_breakout.yaml`) to validate the pipeline on Apple Silicon.
- Add a value-based baseline with our **dueling Double-DQN + prioritized replay** (`configs/dqn_breakout.yaml`) defined in `src/simple_game.algos`.
- Configure **frame stacking** and **gray-scaling** via Gymnasium wrappers; use SB3 `VecFrameStack` and `VecTransposeImage` as needed.
- Implement logging via TensorBoard and periodic model checkpointing so we can resume or compare policies across experiments.
- Record evaluation episodes using `VecVideoRecorder` for qualitative review and archive them in `runs/videos/`.

### Apple Silicon specifics

- Stable-Baselines3's default `NatureCNN` can crash on Apple Silicon when combined with PyTorch's MPS backend. We bundle `src/simple_game/policies.py` with a compatible drop-in replacement (`SimpleNatureCNN`) and wire it up via `policy_kwargs`.
- When writing custom scripts, import `gymnasium` lazily (as done in `train.py`) to avoid an upstream segfault that appears when `gymnasium` is imported before SB3 initialises its policies on MPS.

### Scaling up

- Once PPO baseline works, experiment with **A2C** (faster iteration) or **DQN** (discrete action baseline).  
- Evaluate impact of different preprocessing (frame skip, clip rewards).  
- Add hyperparameter sweeps using Optuna or WandB once manual experimentation plateaus.

## Evaluation & Diagnostics

We share a CLI that evaluates any SB3 checkpoint and optionally wraps the env with additional instrumentation:

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

- `--seeds` adds offsets to the config seed so we can report how metrics vary across evaluation seeds.
- `--deterministic` switches between greedy (deterministic) and sampled (stochastic) action selection. Omit it for stochastic runs.
- `--events` enables `BreakoutEventWrapper`, which logs FIRE presses, first-hit steps, and streak lengths for each life and aggregates game statistics.
- `--output-dir` writes `per_life.csv`, `per_game.csv`, and `summary.json` for downstream analysis (e.g., the dashboard notebook).

Run the command twice per experiment (deterministic / stochastic) to compare performance, then open
`notebooks/breakout_eval_dashboard.ipynb` to visualise serve timing histograms, high-FIRE failures,
and brick-hit streaks.

The evaluator also prints a concise console summary (mean rewards, variance, zero-reward serve failures, high-FIRE retries) so regressions stand out immediately.

See `docs/evaluation_tooling.md` for a deeper dive into the exported CSV/JSON formats and notebook workflow.
