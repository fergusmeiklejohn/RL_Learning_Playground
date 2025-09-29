# Experiments Log

## 2025-02-14 – PPO Baseline on Breakout (M3 Max, MPS)

**Config**: `configs/ppo_breakout.yaml` (`breakout_ppo_baseline`)

**Objective**: Establish a reproducible PPO baseline on `ALE/Breakout-v5` using Stable-Baselines3 and Apple Silicon (MPS) acceleration. Validate the project scaffolding (Conda env, Atari ROM setup, TensorBoard logging, video capture) and confirm that our custom `SimpleNatureCNN` extractor prevents MPS segfaults.

**Setup Highlights**
- Environment pinned via `env/environment.yml` (PyTorch 2.3.0, Gymnasium 0.29.1, SB3 2.3.0, OpenCV 4.9.0.80, NumPy <2).
- Training script: `src/simple_game/train.py`, using 8 parallel envs, PPO defaults adapted from SB3 Atari tuning, `VecFrameStack` + `VecTransposeImage`.
- Custom feature extractor: `src/simple_game/policies.SimpleNatureCNN` to avoid SB3’s default NatureCNN MPS crash.
- Console progress callback + TensorBoard logs at `runs/tensorboard/breakout_ppo_baseline`.
- Checkpoints saved under `runs/checkpoints/breakout_ppo_baseline_*`; evaluation video recorded with `VecVideoRecorder` into `runs/videos/breakout_ppo_baseline/`.

**Run Notes**
- Total timesteps: 2,000,000 (approx. 82 minutes wall clock, ~392 FPS).
- Training metrics stabilized around `ep_rew_mean ≈ 22`, `ep_len_mean ≈ 797` near completion.
- Final SB3 evaluation (5 episodes): mean return 13.40, mean length 588.2 steps (low-sample variance noted; plan to re-evaluate with more seeds/episodes).
- Final evaluation video saved successfully after switching to `VecVideoRecorder` (avoided mismatch shape crash).

**Next Actions**
1. Perform extended evaluation (≥30 episodes, multiple seeds) to quantify variance and confidence intervals.
2. Inspect TensorBoard curves for reward stability, KL trends, value loss, and entropy decay; log observations in `docs/`.
3. Analyze recorded gameplay to identify qualitative weaknesses (serve handling, paddle positioning).
4. Prepare follow-up experiments: learning-rate schedules, entropy tuning, or algorithm comparisons (DQN/A2C) once baseline is understood.
