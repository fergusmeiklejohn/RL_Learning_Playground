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

## 2025-09-30 – Extended Evaluation Pass

**Artifacts reviewed**: `runs/tensorboard/PPO_3` (latest PPO baseline), `runs/checkpoints/breakout_ppo_baseline_*.zip`, `runs/videos/breakout_ppo_baseline/eval-step-0-to-step-2000.mp4`.

**TensorBoard takeaways**
- `rollout/ep_rew_mean` climbs from ~1.6 to a peak of 25.6 by 1.95M steps; the curve is smooth without oscillations, suggesting stable improvement during training.
- `rollout/ep_len_mean` follows the reward trend (up to 880 steps) and `train/approx_kl` stays mostly in the 0.02–0.07 band (occasional spikes to 0.18), well below the PPO clip threshold.
- `train/value_loss` decays from 0.48 → 0.25, while `train/entropy_loss` drops to ~-0.33, indicating the policy is confidently selecting actions late in training.

**Fresh evaluation (30 episodes, deterministic unless noted)**
- Script: load config → build eval env with the same wrappers → `PPO.load` checkpoint on CPU → collect statistics. (Note: importing NumPy before `PPO.load` still segfaults on MPS; load first, then import.)
- **Per-life view** (treat each life as an episode, matching the `LifeLoss` wrapper output): deterministic mean reward **3.71 ± 3.14**, min/max 0–12; mean length **38.8 ± 29.6** steps. This perspective highlights instability—many serves still fail immediately.
- **Per-game view** (use `info['episode']` like SB3’s Monitor/TensorBoard): deterministic mean reward **22.83 ± 6.87**, mean length **847 ± 163** steps; stochastic sampling drops to **≈21.2 ± 6.9**. These numbers align with the tensorboard `rollout/ep_rew_mean ≈ 20.5` and clarify why greedy play outperforms stochastic sampling.
- Key insight: TensorBoard and `evaluate_policy` aggregate full games via the Monitor’s `episode` info, whereas naïvely summing rewards at each `done` event double-counts individual lives. Our early script was in the latter camp, which is why it under-reported performance.
- Added CLI helper `python -m src.simple_game.evaluate` to print both views (and SB3-style stats) without importing NumPy before `PPO.load`; it accepts seed offsets, device overrides, an `--events` flag, and exports CSV/JSON summaries when `--output-dir` is provided.
- Per-life CSV (`runs/eval_reports/*/per_life.csv`) now logs reward, length, FIRE presses, first FIRE/brick-hit steps, and max positive streak per life—useful for spotting serve failures and tracking how long the paddle survives before scoring.
- Per-game CSV/summary capture cumulative reward, episode length, lives played, total brick hits, FIRE usage, and the share of lives that ever score; `summary.json` bundles the aggregated stats for automation.
- Multi-seed sweep (seed offsets 10000/10001/10002):
  - Deterministic per-game means **22.83 / 22.47 / 23.57** (aggregate **22.96 ± 0.56**), positive-life ratio ≈ **0.82**, avg FIRE presses ≈ **49** per game.
  - Stochastic per-game means **21.40 / 22.53 / 19.57** (aggregate **21.17 ± 1.50**), showing a consistent drop when sampling actions, even though positive-life ratios remain within **0.68–0.84**.
- Failure clusters: across deterministic seeds **58/450** lives ended with zero reward (mean length ~3 frames); only **4** of those lives attempted ≥3 FIRE presses before losing, so most failures are instant misses right after the serve. Stochastic runs increase both zero-reward lives and high-FIRE retries (up to 6 lives ≥3 FIRE presses per seed), hinting that sampled actions often jitter during serve recovery.
- Notebook `notebooks/breakout_eval_dashboard.ipynb` loads the exported CSVs and plots the distribution of `first_positive_step` values alongside headline ratios for quick visual inspection.

**Open questions / follow-ups**
1. Use the exported CSVs to dig into failure clusters (e.g., lives with many FIRE presses but zero reward) and feed insights back into environment tweaks or curriculum ideas.
2. Consider logging/visualising brick-hit streaks over time (line plots per episode) to see whether the agent stalls mid-volley.
3. Run the enhanced evaluator on at least three fresh seeds for every future experiment so we can compare distributions, not just means.
4. Once evaluation tooling is routine, branch a new config for an alternative algorithm (e.g., DQN with dueling network) for comparison.

## 2025-09-30 – Dueling Double-DQN Kickoff (in progress)

**Config**: `configs/dqn_breakout.yaml` (`breakout_dqn_dueling`)

**Objective**: Establish a value-based baseline that complements PPO by training a dueling Double-DQN with prioritized replay. We want to evaluate how a replay-based learner handles the same serve failures surfaced in the PPO analysis and whether it achieves steadier improvement under exploration decay.

**Key Setup**
- Algorithm: custom `PrioritizedDQN` (Double-DQN updates, dueling Q-network). Policy is `src.simple_game.algos.DuelingCnnPolicy` (dueling head + `SimpleNatureCNN` features).
- Replay: proportional prioritized buffer (`alpha=0.6`, `beta0=0.4`, epsilon `1e-6`) with 100k capacity, 50k warm-up steps, updates every 4 env steps.
- Exploration: linear schedule from 1.0 → 0.01 over 10% of training; remaining steps exploit learned policy.
- Training budget: 3,000,000 environment steps (~4–5 hours wall clock on M3 Max); checkpoints every 500k steps, evaluations every 100k (5 deterministic episodes), videos enabled.
- Logging: `runs/tensorboard/DQN_*/`, checkpoints in `runs/checkpoints/`, monitor traces under `runs/monitor/train/breakout_dqn_dueling`.

**Status (2025-09-30 @ ~53k steps)**
- FPS ≈ 450 on MPS with a single env; exploration rate still ≈0.82, reward curve slowly climbing from 0 to ~1.8 as replay fills.
- Loss steadily decreasing (<0.01) once replay warms up; prioritized sampling focuses on high-TD-error transitions (serve failures currently dominate).
- No stability issues observed: prioritized buffer integrates cleanly with custom weight updates, and dueling head runs without MPS quirks.

**Final results (2025-10-01)**
- Training completed 3,000,000 steps (~5h22m). SB3’s terminal eval (5 episodes) reported **31.4 ± 16.7** reward, **1039** mean length.
- Extended evaluation (30 games × 3 seeds):
  - Deterministic mean reward **33.8 ± 3.2** (per-life mean **4.7**), with only **16–25 zero-reward lives** per seed (4–6× fewer than PPO) and no instances of ≥3 FIRE presses without scoring.
  - Stochastic mean reward **35.7 ± 1.5**, demonstrating the policy remains strong under sampling (contrast to PPO, which lost ~2.3 points when stochastic).
  - Game lengths averaged **~1,020** frames, highlighting longer volleys and improved serve recovery.
- Serve analysis: majority of lives now score at least once; failure clusters from PPO (instant misses) dropped sharply, confirming prioritized replay + dueling head help the agent stabilise the opening volley.
- Recorded evaluation video rebuilt via the new `imageio` pipeline (`runs/videos/breakout_dqn_dueling/eval-step-0-to-step-2000.mp4`).

**Next checkpoints**
1. Let the run finish the full 3M steps (ETA ~4–5h). Keep TensorBoard open to monitor reward/exploration curves.
2. After completion, run the evaluation CLI with `--events` for both deterministic and stochastic settings to capture per-life/per-game diagnostics. Compare zero-reward serve failures vs PPO baseline.
3. Update this log with final metrics (30-game mean reward, per-life stats, exploration schedule behaviour) and note whether prioritized replay mitigated the immediate-serve-drop issue.
4. Depending on results, explore Rainbow-style additions (noisy nets, n-step returns) or adjust buffer size / exploration schedule for faster ramp-up.
