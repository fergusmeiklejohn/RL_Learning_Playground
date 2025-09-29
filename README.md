# Simple Atari RL Playground

This project aims to train reinforcement learning agents on classic Atari games using [Gymnasium](https://gymnasium.farama.org/) environments and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) on an Apple Silicon (M3 Max) Mac running macOS. The goal is to keep the workflow transparent by documenting decisions, trade-offs, and alternatives.

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
   - Iterate on hyperparameters or algorithms (e.g., A2C, DQN) once baseline PPO is stable.

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

- Start with **PPO**: stable on Atari, robust hyperparameter defaults, and supports vectorized environments for throughput.
- Configure **frame stacking** and **gray-scaling** via Gymnasium wrappers; use SB3 `VecFrameStack` and `VecTransposeImage` as needed.
- Implement logging via TensorBoard and periodic model checkpointing.  Checkpoints allow resuming training and comparing policies across experiments.
- Record evaluation episodes using Gymnasium wrappers with `RecordVideo` for qualitative review.

### Apple Silicon specifics

- Stable-Baselines3's default `NatureCNN` can crash on Apple Silicon when combined with PyTorch's MPS backend. We bundle `src/simple_game/policies.py` with a compatible drop-in replacement (`SimpleNatureCNN`) and wire it up via `policy_kwargs`.
- When writing custom scripts, import `gymnasium` lazily (as done in `train.py`) to avoid an upstream segfault that appears when `gymnasium` is imported before SB3 initialises its policies on MPS.

### Scaling up

- Once PPO baseline works, experiment with **A2C** (faster iteration) or **DQN** (discrete action baseline).  
- Evaluate impact of different preprocessing (frame skip, clip rewards).  
- Add hyperparameter sweeps using Optuna or WandB once manual experimentation plateaus.

## Next Steps

- Fill in `configs/` with initial PPO hyperparameters tailored to Breakout.  
- Flesh out `src/simple_game/train.py` with CLI and training loop.  
- Add evaluation script to visualize agent behavior.  
- Document results and insights in `docs/`.
