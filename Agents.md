# Session Notes: Simple Atari RL Playground

## Overview
This repo is a vehicle for the user to learn RL. Please explain the process, the terms used and the code, and the user will learn by doing and observing. By the end of this session, the user should understand how to train a simple Atari agent using different methods, and understand the key decisions made, evaluations, and actions taken.

## Process
We will create and keep updated an experiments.md file to track the training and evaluation experiments we run, the parameters and/or methods used, and the results obtained. This will help us keep track of what works and what doesn't.

## Environment & Tooling
- Always use the project Conda environment `simple-game`; update (instead of recreating) with `conda env update -n simple-game -f env/environment.yml --prune`.
- When you need Python directly from the env (outside `conda run`), call `/Users/fergusmeiklejohn/miniconda3/envs/simple-game/bin/python`.
- After editing `env/environment.yml`, re-run the update command above so that pip deps (PyTorch, Gymnasium extras, AutoROM, OpenCV, rich) stay in sync.

## Atari Training Pipeline
- Entry point: `python -m src.simple_game.train --config configs/ppo_breakout.yaml`.
- Configs live under `configs/`; tweak `policy_kwargs`, `hardware.device`, or hyperparameters there rather than in code.
- Training scaffolding builds both train/eval envs with `VecFrameStack` + `VecTransposeImage` and logs to `runs/`.
- Use `SimpleNatureCNN` (`src/simple_game/policies.py`) – the default SB3 `NatureCNN` segfaults on Apple Silicon due to PyTorch MPS tensor conversion.
- Keep Gymnasium imports lazy (as in `train.py`) to dodge another upstream segfault on macOS/MPS.

## Debugging & Observability
- Progress appears in the shell via the custom console callback plus SB3’s built-in progress bar.
- TensorBoard logs write to `runs/tensorboard/<experiment>`; evaluations/checkpoints land in `runs/checkpoints`.
- Recordings (when enabled) save to `runs/videos/<experiment>`.

## Miscellaneous
- AutoROM expects the ROM directory to exist (`~/.gymnasium/atari_roms`); the setup script already handles this.
- If new dependencies are needed mid-session, prefer `conda env update ... --prune` rather than deleting the env.
- The repo now exposes `simple_game.policies` via `__all__`, so `policy_kwargs` can reference `src.simple_game.policies.SimpleNatureCNN` safely.
