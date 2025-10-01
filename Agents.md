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

## Progress To Date
- Completed baseline PPO run (`configs/ppo_breakout.yaml`) at 2M timesteps. Result: modest survival improvements but the agent fails to score consistently because PPO dropped replayed signal from sparse rewards.
- Completed dueling Double-DQN run (`configs/dqn_breakout.yaml`) at 3M timesteps with prioritized replay. Result: longer rallies and higher mean rewards than PPO, yet still no reliable game wins at this training budget.
- Key takeaway: off-policy replay plus dueling advantages accelerate learning on sparse Breakout rewards, but data requirements remain high when working from raw pixels and primitive actions.

## Upcoming Exploration: Injecting Structure
- **Object-centric encoders**: build or integrate feature extractors that detect distinct entities (paddle, ball, bricks) before policy learning to provide symbolic inputs.
- **Hierarchical / options-based control**: learn mid-level skills (e.g., tracking the ball, setting up tunnels) and a controller that chooses among them to shorten credit assignment chains.
- **Model-based planning**: experiment with world-model agents (e.g., MuZero-style or Dreamer variants) that learn dynamics and plan or imagine trajectories instead of relying purely on Monte Carlo rollouts.
- **Imitation and curricula**: seed policies from demonstrations or staged tasks to expose higher-level strategies earlier than epsilon-greedy exploration can discover alone.
- **Structured observations**: replace or augment pixel input with RAM/state taps or external object trackers to compare how much abstraction reduces sample complexity.

These strands will each get dedicated experiments so we can examine how they work, where they break, and what tooling or theory is needed to push them further.

### Current Focus: Detector-Augmented Object Encoders
- Starting with detector-augmented object-centric encoders because Breakout’s visuals make deterministic paddle/ball/brick tracking straightforward (color thresholds or RAM taps), giving us fast symbolic features.
- Expectation is that providing explicit entities will highlight the benefits of abstraction before we invest in heavier unsupervised slot-attention training.
- Outcomes from this variant will act as the baseline for later slot-attention experiments so we can quantify the trade-off between engineered detectors and learned object discovery.
