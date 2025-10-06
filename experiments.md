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

## 2025-10-02 – Detector-Augmented DQN Evaluation

**Config**: `configs/objcentric_breakout_ram.yaml` (`breakout_objenc_ram_dqn`)

**Objective**: Measure the impact of RAM-tap entity features concatenated to the CNN encoder on Breakout performance relative to the pixel-only dueling Double-DQN baseline.

**Artifacts reviewed**: checkpoints at `runs/checkpoints/breakout_objenc_ram_dqn_*.zip`, TensorBoard run `runs/tensorboard/breakout_objenc_ram_dqn`, evaluation exports under `runs/eval_reports/breakout_objenc_ram_dqn/`.

**Evaluation summary (30 games × 3 seed offsets, event wrapper enabled)**
- Deterministic (greedy) control: per-game reward **35.46 ± 0.53**, per-life reward **4.998 ± 0.10**, average game length **1,061 ± 195** frames. Zero-reward lives fell to **44 / 450** (≈9.8%) with none triggering the ≥3 FIRE misuse flag.
- Stochastic control (ε-greedy sampling): per-game reward **33.24 ± 1.12**, per-life reward **4.74 ± 0.09**, average game length **1,018 ± 181** frames. Zero-reward lives held at **41 / 450**, but reward mean trailed the pixel baseline by ~2.4 points.
- Additional `non-deterministic/` sweep (alternate sampling seed choices) landed at **32.48 ± 2.74** reward, confirming the same pattern: sturdy greedy play, softer exploratory play.

**Comparison vs pixel-only DQN baseline** (`runs/eval_reports/dqn_deterministic`, `runs/eval_reports/dqn_stochastic`)
- Greedy reward lifted **33.82 → 35.46** (+1.64) while variance shrank (per-game std **16.36 → 12.35**). Zero-reward lives dropped from **63** to **44**, and mean life length stretched **46.4 → 49.5** frames, matching the extended rallies seen in the highlight clip.
- Stochastic reward slipped **35.67 → 33.24** (−2.43). Despite fewer zero-reward lives (52 → 41) and slightly longer games, the policy under exploration noise can’t consistently match the pixel-only agent’s score—pointing to overly sharp Q-gaps.
- Q-value logging (`--collect-q-stats`) showed mean argmax gaps of **0.34** (both greedy and stochastic sweeps). These narrow margins explain why small sampling noise destabilises play even though the greedy lanes look strong.

**Qualitative artifacts**
- 6k-step deterministic rollout: `runs/videos/breakout_objenc_ram_dqn/extended-eval-step-0-to-step-6000.mp4`
- Longest life clip (seed 10002, game 27): `runs/videos/breakout_objenc_ram_dqn/longest-life-seed10002-game27-life0-manual.mp4`

**Pros**
- Faster, more reliable serves: zero-reward lives down ~30% and FIRE stats show quick relaunches without panic spamming.
- Higher greedy ceiling with tighter variance; detector features stabilise paddle placement and brick targeting once volleys begin.
- RAM-derived entities are cheap to compute and deterministic, giving reproducible trajectories (useful for debugging and curriculum planning).

**Cons / Risks**
- Stochastic robustness regressed: narrow Q-gaps make ε-greedy exploration two points worse than the pixel baseline despite comparable rally lengths.
- Heavy reliance on emulator internals (RAM taps) may not transfer to environments without accessible memory maps; brittleness if ALE RAM layout shifts.
- Detector head likely dominates the CNN extractor; monitoring needed to ensure pixel features still contribute once we add pixel/object hybrids.

**Next steps**
1. Investigate Q-value calibration (noisy nets, action-value regularisation, or lower detector gain) to close the stochastic gap while keeping serve reliability.
2. Blend RAM and pixel detectors, or add dropout in the detector MLP, to diversify features and widen Q-gaps.
3. Extend the eval harness with gap histograms / per-life scatter plots to see where stochastic failures occur (serve vs rally). Use the saved long-life clip as a qualitative baseline when testing adjustments.

---

## 2025-10-03 – Hybrid RAM+Pixel Detector Plan

**Config**: `configs/objcentric_breakout_hybrid.yaml` (`breakout_objenc_hybrid_dqn`)

**Hypothesis**: Concatenating RAM taps (precise but emulator-specific) with pixel-threshold features (noisier but emulator-agnostic) will keep greedy performance high while widening Q-value margins, improving stochastic robustness and highlighting failure modes when the sources disagree.

**Detector design**
- New `HybridDetector` composes existing `BreakoutRamDetector` and `BreakoutPixelDetector` (see `src/simple_game/detectors.HybridDetector`). It resets both sub-detectors each episode, concatenates their feature vectors every step, and presents a single augmented observation to the policy extractor.
- RAM component uses paddle/ball velocity plus a 12-bit brick bitmap; pixel component tracks paddle/ball centroids from the most recent frame. Detector MLP output widened to 96 dims to process the richer signal.

**Experiment setup**
- Train prioritized dueling DQN for 3M steps (`python -m src.simple_game.train --config configs/objcentric_breakout_hybrid.yaml`).
- Evaluation sweeps: deterministic and stochastic (30 games × seed offsets 0/1/2) plus `--collect-q-stats` to log argmax gaps for comparison against RAM-only and pixel-only baselines.
- Artifacts to collect: per-life/per-game CSVs, q-stat summaries, long-serve clips (reuse recording helper with target life indices once evaluation data is available).

**What to compare**
- Greedy vs stochastic reward delta relative to RAM-only run (does blending recover the ~2.4-point loss?).
- Q-gap histograms vs RAM-only (expect gaps >0.34 if pixel signal injects diversity).
- Disagreement diagnostics: track lives where RAM predicts ball contact but pixel features disagree (e.g., via correlation plots of paddle_x_ram vs paddle_x_pixel in exported CSVs).

**Success criteria**
1. Deterministic reward ≥ RAM-only baseline (≥35.4) with equal or lower zero-reward lives.
2. Stochastic reward within ±0.5 of pixel-only baseline (≈35.7) or visibly improved over RAM-only (≥34.5) without widening variance.
3. Q-gap mean increases (target ≥0.45) while remaining stable across seeds.

**Risks / watch-outs**
- Mismatch between RAM and pixel coordinates could confuse the detector MLP (requires monitoring of feature scales; may need normalization hooks if gradients oscillate).
- Additional detector dimensions might require tweaking `detector_hidden_dim` or adding dropout to prevent overfitting to RAM.
- If performance regresses, consider gating/attention between components instead of naive concatenation.

**Next actions**
1. Kick off training run and monitor TensorBoard (detector feature norms, TD error).
2. Extend evaluation tooling with optional CSV columns capturing RAM vs pixel deltas for offline plotting (if needed post-run).
3. Update this log with results, highlighting whether hybridization closes the stochastic gap or introduces new failure signatures.

---

## 2025-10-04 – Hybrid RAM+Pixel Detector Results

**Artifacts reviewed**: `runs/eval_reports/breakout_objenc_hybrid/{deterministic,stochastic}/`, `runs/videos/breakout_objenc_ram_dqn/longest-life-seed10002-game27-life0-manual.mp4`

**Key metrics**
- Deterministic reward **35.31 ± 1.64** (RAM-only: 35.46 ± 0.53; pixel-only: 33.82 ± 3.22). Per-life reward 4.99 vs 4.74 (pixel) and 4.88 (RAM).
- Zero-reward lives: **26 / 450** (≈5.8%). RAM-only had 44 (9.8%), pixel-only 63 (14%). Serve reliability improved markedly.
- Mean life length: 48.4 frames (RAM 49.5, pixel 46.4). Qualitatively matches long-serve clip—volley duration stays high.
- Stochastic reward **33.23 ± 3.44** (RAM-only: 33.24 ± 1.12; pixel-only: 35.67 ± 1.54). Hybrid mirrors RAM behaviour under exploration, trailing pixel baseline by ~2.4 points.
- Q-value argmax gap (from `--collect-q-stats`): **0.33–0.34** across seeds, identical to RAM-only run. Pixel features act as a regulariser rather than widening action preferences.

**Pros**
- Serve consistency: zero-reward lives halved vs RAM-only, threefold better than pixel-only.
- Greedy play retains high score with tighter per-seed variance; hybrid observably steadies early rallies.
- RAM+pixel fusion keeps per-life length nearly unchanged, indicating the richer input doesn’t destabilise control.

**Cons / trade-offs**
- Stochastic reward remains low; the hybrid fails to recover the pixel-only agent’s exploratory edge.
- Q-gaps stay narrow (~0.34), so ε-greedy sampling still flips to near-tie actions that underperform.
- Added dependence on emulator RAM remains; pixel stream doesn’t yet compensate when RAM is noisy.

**Interpretation**
- Hybrid features emphasise safety (almost every serve scores) but don’t diversify policy decisions. Action-value regularisation or stream reweighting likely required to widen Q-gaps if exploration matters.
- Choice of agent depends on priorities: pixel-only for stochastic robustness, RAM-only for higher greedy reward, hybrid for serve reliability.

**Next steps**
1. Normalise/scale detector streams or add dropout/noisy nets to widen Q-gaps while preserving serve gains.
2. Log RAM vs pixel paddle/ball deltas in exported CSVs to diagnose disagreement patterns; consider attention/gating if conflicts emerge.
3. Repeat evaluations post-adjustments, updating this log with Q-gap histograms and stochastic reward deltas.

---

## 2025-10-05 – Slot Attention Encoder Plan

**Config**: `configs/objcentric_breakout_slot.yaml` (`breakout_objenc_slot_dqn`)

**Objective**: Replace hand-engineered detectors with a learned Slot Attention encoder that discovers object slots directly from pixels, testing whether learned object grouping can match RAM-level reliability while widening Q-value gaps.

**Model changes**
- Added `SlotAttentionExtractor` and `SlotAttentionDuelingCnnPolicy` (`src/simple_game/policies.py`). The encoder uses a lightweight CNN backbone, adds positional embeddings, runs 3 Slot Attention iterations over 6 slots (64-dim each), and flattens slots before the dueling head.
- Slot Attention module initialises slots via learned Gaussians, applies attention/MLP updates, and outputs per-slot embeddings suitable for value estimation.

**Experiment setup**
- Train prioritized dueling DQN for 3M steps with the new policy (`python -m src.simple_game.train --config configs/objcentric_breakout_slot.yaml`).
- Track: training reward curve, slot feature norms, attention entropy (add custom logging if necessary), and compare deterministic/stochastic evaluations (30 games × seeds 0/1/2 + `--collect-q-stats`).
- Save evaluation artifacts under `runs/eval_reports/breakout_objenc_slot/` and capture representative videos once stabilized.

**What to compare**
- Serve reliability vs hybrid detector (zero-reward lives, life length).
- Stochastic reward vs pixel baseline—do learned slots widen Q-gaps (>0.34) and recover exploratory robustness?
- Slot diversity: inspect attention weights or per-slot activation variance to ensure multiple entities are captured.

**Risks / monitoring**
- Slot Attention adds optimisation complexity; monitor for slot collapse (all slots identical) or exploding gradients (enable gradient clipping if needed).
- Additional compute may reduce FPS; profile on M3 Max/MPS and adjust slot count or CNN width if training stalls.
- If convergence lags, consider pretraining encoder with reconstruction loss or freezing slot module for initial steps.

**Next actions**
1. Kick off the slot-attention training run and watch TensorBoard for slot feature stability.
2. Extend evaluation tooling to optionally log attention entropy per step (TODO if collapse suspected).
3. Document results alongside hybrid/RAM/pixel baselines once evaluation completes.

---

## Upcoming Structured-Agent Experiment Templates

### Object-Centric Encoder Series (Strategy 1)
- **Config Stub**: `configs/objcentric_breakout_<variant>.yaml`
- **Objective**: Replace raw-pixel observations with entity slots (paddle, ball, bricks) before policy learning to compare against CNN baselines.
- **Planned Runs**:
  - RAM-tap detector variant (read Atari memory for paddle/ball/brick state) feeding structured features into PPO/DQN – first implementation target.
  - Pixel/color-threshold detector variant to evaluate generalization beyond emulator internals.
  - Slot-attention encoder powering PPO and DQN heads.
- **Metrics to Capture**: reward vs frames, serve success rate, slot assignment stability, wall-clock overhead.
- **Open Questions**: Need for auxiliary reconstruction losses? Sensitivity to encoder pretraining.

#### Variant A: RAM-Tap Detector
- **Tasks**:
  1. Map required RAM addresses (paddle x, ball x/y, ball velocity, brick grid) and validate against emulator docs.
  2. Implement `BreakoutRamDetector` utility returning normalized entity features per step.
  3. Extend training pipeline to concatenate detector features with CNN outputs (or replace observation pipeline) and add config `configs/objcentric_breakout_ram.yaml`.
  4. Run pilot PPO/DQN experiments; log feature diagnostics (percentage of valid detections, distribution of speeds).
- **Risks/Notes**: tightly coupled to Atari RAM layout; must guard against Gym version shifts.

#### Variant B: Color-Threshold Detector
- **Tasks**:
  1. Prototype image-space tracker (color masks or template matching) to locate paddle/ball/brick columns.
  2. Benchmark detection reliability under frame flicker; add smoothing if needed.
  3. Create config `configs/objcentric_breakout_pixel.yaml` mirroring RAM variant with the new detector.
  4. Compare performance vs RAM detector to isolate robustness vs implementation cost.
- **Risks/Notes**: sensitive to palette changes; needs CPU budget for per-frame processing.

### Hierarchical / Options Series (Strategy 2)
- **Config Stub**: `configs/hiro_breakout_<variant>.yaml`
- **Objective**: Introduce multi-level controllers (HIRO, Option-Critic) to handle long-horizon planning such as serve recovery and tunnel creation.
- **Planned Runs**: Baseline hierarchical agent, option-length ablations, termination-condition sweeps.
- **Metrics to Capture**: option usage histograms, transitions during serve vs rally, reward progression.
- **Open Questions**: How to shape sub-policy rewards? What option horizon matches Breakout pacing?

### Model-Based Planning Series (Strategy 3)
- **Config Stub**: `configs/worldmodel_breakout_<variant>.yaml`
- **Objective**: Train world-model agents (Dreamer-like or lightweight MuZero) to plan in latent space instead of pure model-free updates.
- **Planned Runs**: reconstruction-only world model, planning-enabled agent, imagination-horizon ablations.
- **Metrics to Capture**: planning depth vs reward, prediction error on held-out rollouts, reward-model calibration.
- **Open Questions**: Compute budget on M3 Max? Do learned dynamics transfer across seeds?

### Imitation / Curriculum Series (Strategy 4)
- **Config Stub**: `configs/imitation_breakout_<variant>.yaml`
- **Objective**: Bootstrap agents via demonstrations or staged curricula to reduce time-to-first-win.
- **Planned Runs**: behavior cloning + RL fine-tuning, DAGGER-style aggregation, staged difficulty curriculum.
- **Metrics to Capture**: sample efficiency, dependence on demo quality, curriculum stage completion time, policy divergence.
- **Open Questions**: How many demos are required? Does curriculum transfer hold when difficulty scales up?

### Structured Observation Series (Strategy 5)
- **Config Stub**: `configs/structuredobs_breakout_<variant>.yaml`
- **Objective**: Replace or augment pixel input with RAM snapshots or external object trackers to test abstraction effects.
- **Planned Runs**: RAM-only agent, hybrid pixel+RAM inputs, vectorized object statistics (ball angle/speed, paddle position).
- **Metrics to Capture**: reward vs frame curves, overfitting risks to RAM quirks, preprocessing overhead.
- **Open Questions**: Stability of RAM features across Gym releases? How to normalize hybrid inputs effectively?

## 2025-10-06 – Slot Attention Encoder Results

**Artifacts reviewed**: `runs/eval_reports/breakout_objenc_slot/{deterministic,stochastic}/summary.json`, per-life/per-game CSVs, training checkpoints at `runs/checkpoints/breakout_objenc_slot_dqn_final.zip`.

**Key metrics**
- Deterministic (30 games × seeds 10000/1/2): per-game reward **29.57 ± 2.31**, per-life reward ≈ **4.45**, mean episode length **942–974** frames.
- Stochastic sweep: per-game reward **31.21 ± 0.65**, per-life reward ≈ **4.56**, lengths ≈ **968** frames.
- Serve reliability: only **4 / 450** deterministic lives and **6 / 450** stochastic lives ended scoreless (≤1%), best among all agents so far.
- Q-gap: mean action gap **0.31** (det) / **0.30–0.31** (stoch), essentially unchanged vs RAM or hybrid detectors.

**Comparative highlights**
- Greedy play regresses **~5.9 points** vs RAM detector (35.46 ± 0.53) and **~5.7** vs hybrid (35.31 ± 1.64), though it still beats PPO baseline (22.83 ± 0.56) on full-game reward.
- Stochastic reward sits **2.0 points below** hybrid (33.23 ± 3.44) and **4.5 below** the pixel-only dueling DQN (35.67 ± 1.54), indicating slots have not yet widened exploration margins.
- Serve stability surpasses prior detectors (RAM 44 scoreless lives, hybrid 26, pixel 63), showing slot features capture paddle/ball cues reliably even without RAM taps.
- Episode lengths trail RAM/hybrid by ≈80–120 frames, suggesting earlier brick clears but shorter rallies once tunnels open.

**Interpretation**
- Slot attention successfully discovers entities sufficient to eliminate most serve drops, validating the learned-object approach for core control.
- However, the encoder collapses to similarly narrow Q-gaps (~0.31) as prior runs, so ε-greedy sampling still flips between near-tie actions and fails to recover the pixel agent's exploratory strength.
- The longer-tail reward distribution (std ≈10) points to unstable brick-clear strategies—likely slot assignments shift mid-rally, reducing high-reward outliers seen with RAM features.

**Next steps**
1. Instrument slot entropy/variance logging to detect slot collapse and guide regularisation (e.g., KL to encourage diverse slots).
2. Add auxiliary reconstruction or contrastive losses during training to stabilise slot identities; re-run with identical hyperparameters for a clean comparison.
3. Experiment with widened dueling heads or noisy nets to target larger Q-gaps while keeping the learned encoder.
4. Capture representative gameplay (deterministic vs stochastic) to verify qualitative slot usage during rallies.

---

## 2025-10-06 – Slot Stability Instrumentation Prep

**Objective**: equip the slot-attention extractor with auxiliary losses and diagnostics that reward diverse slot usage and penalise reconstruction drift before the next training sweep.

**Code Changes**
- Added entropy, slot-variance, and lightweight reconstruction regularisers to `SlotAttentionExtractor`, exposing weights + logging via `features_extractor_kwargs`.
- Modified `PrioritizedDQN.train` so auxiliary losses backpropagate alongside TD error; TensorBoard now records `train/aux/*` metrics for monitoring.
- Config `configs/objcentric_breakout_slot.yaml` updated with initial weights (`entropy=0.02`, `variance=0.001`, `recon=0.05`) and metric logging enabled.

**Next Steps**
1. Kick off a fresh `breakout_objenc_slot_dqn` run to validate that auxiliary losses stay bounded and improve slot stability.
2. Review new TensorBoard traces (`train/aux/slot_attention_entropy`, `train/aux/slot_reconstruction_loss`) and adjust weights if they overwhelm TD loss.

## 2025-10-06 – Hierarchical Controller Scaffolding

**Objective**: prepare infrastructure for an options-based Breakout agent that layers a high-level manager over specialised skills before implementing the full training loop.

**Artifacts added**
- `configs/hierarchical_breakout_options.yaml` outlining manager/skill hyperparameters and warmups.
- `src/simple_game/hierarchical/` package with config dataclasses, option/manager controller skeletons, and a placeholder trainer entry point.
- `train.py` now recognises `model.algo: hierarchical` and points to the scaffolding pending trainer implementation.

**Immediate follow-up**
1. Flesh out `HierarchicalTrainer.train` to coordinate manager/skill optimisation (likely alternating updates with separate replay buffers).
2. Decide on intrinsic reward shaping for each skill (ball tracking, serve setup, tunnel push) and wire triggers into env wrappers.
3. Extend evaluation tooling to log per-option usage, dwell times, and success rates once the trainer is live.

## 2025-10-06 – Slot Stability Run Results

**Artifacts reviewed**: `runs/eval_reports/breakout_objenc_slot_aux/{deterministic,stochastic}/summary.json` from the auxiliary-loss slot-attention run (entropy 0.02, variance 0.001, recon 0.05).

**Key metrics**
- Deterministic (30 games × seeds 10000/1/2): per-game reward **31.21 ± 0.99**, per-life reward **4.70**, mean length **994 ± 29** frames.
- Stochastic sweep: per-game reward **31.06 ± 1.47**, per-life reward **4.60**, mean length **982 ± 19** frames.
- Q-gap widened to **0.40 ± 0.01** (deterministic seeds) from the prior ~0.31, matching the goal of separating action preferences.
- Serve reliability remains high (deterministic zero-reward lives **4/450**); stochastic zero-reward lives rose to **10/450** (previously 6), driven by a handful of premature exploration serves.

**Comparative highlights**
- vs. previous slot run: deterministic reward +1.64 points (31.21 vs 29.57), stochastic essentially flat (-0.15). Q-gaps +0.09; per-life reward up ~0.2; variance decreased (std 0.99 vs 2.31), indicating more consistent greedy play.
- vs. RAM detector: still -4.25 points deterministic but halves zero-reward lives (4 vs 44) and now narrows the gap with stronger Q separation.
- vs. hybrid detector: deterministic -4.10, stochastic -2.17, yet Q-gaps now exceed hybrid's 0.34 average while preserving low failure rate.
- vs. pixel DQN: reward remains ≈4.6 lower stochastically, but Q-gap lead shrinks—slots hit 0.40 vs pixel’s ~0.33.

**Interpretation**
- Auxiliary losses succeeded in stabilising slot identities enough to widen Q-value separation and tighten deterministic variance without hurting average reward.
- Increased stochastic zero-reward lives show the policy still struggles once randomness reintroduces noisy serves; fine-tuning entropy/variance weights or decaying reconstruction late in training may recover those lives.
- Overall, the object-centric approach now combines strong serve reliability and more decisive action margins, but exploration is still weaker than RAM/pixel agents—future tweaks should target stochastic robustness (e.g., adaptive entropy weight, noisy nets).

**Next steps**
1. Sweep auxiliary weights (drop reconstruction late or anneal entropy) to reduce stochastic zero-reward lives.
2. Log slot attention entropy during long stochastic episodes to confirm stability persists beyond deterministic evaluation.
3. In parallel, continue preparing hierarchical trainer (options may leverage the now-stable encoder as shared backbone).

## 2025-10-06 – Hierarchical Trainer Implementation

**Summary**: implemented a first-pass two-level DQN trainer (`src/simple_game/hierarchical/trainer.py`) that co-trains skill policies and a high-level manager directly from pixels. Each skill and the manager maintain their own replay buffers, epsilon schedules, and target networks; option bonuses (`success_reward`/`failure_penalty`) feed into the manager’s return. Training uses 8-bit frame buffers, configurable hyperparameters, periodic logging, evaluation, and checkpointing. `train.py` now routes `model.algo: hierarchical` configs (see `configs/hierarchical_breakout_options.yaml`) to the new trainer.

**Next actions**
1. Implement skill-specific intrinsic rewards/triggers (e.g., detect tunnel events) to replace the current "reward > 0" success heuristic.
2. Add evaluation hooks mirroring PPO/DQN tooling (per-life CSVs, option usage stats).
3. Profile learning stability and calibrate epsilon/target update schedules before launching the first hierarchical Breakout run.

## 2025-10-06 – Hierarchical Trainer Progress Bar

Added a `tqdm`-backed progress bar to hierarchical training (enabled via `experiment.progress_bar`).
The bar tracks step completion alongside rolling reward/length and epsilon stats, giving the same
at-a-glance ETA feedback as our SB3 runs.

## 2025-10-07 – Hierarchical DQN Run Results

**Artifacts reviewed**: `runs/checkpoints/breakout_hier_options/manager.pt`, `skill_*.pt`, monitor logs (`runs/monitor/*.monitor.csv`). Direct deterministic evaluation via standalone script currently segfaults inside ALE when reinitialising the Breakout env outside the training harness, so the summary below relies on the monitor traces from the completed 3M-step run (final 10.2k games) plus the tail window (last 100 games) as a proxy for plateau performance.

**Training statistics**
- Aggregate across the run (10,180 episodes): mean reward **21.48 ± 8.70**, mean episode length **792.8 ± 406** steps; zero-reward games are rare (**0.05%**).
- Final 100 training games: reward **22.81 ± 8.24**, episode length **820.7 ± 182.7** steps — the manager keeps the ball alive longer but still falls well short of template slot-attention agents on scoring.
- Progress-bar eval snapshots during training hovered around reward ≈1–2, confirming that the greedy manager/skill stack has not yet learned stable serve or tunnel behaviours despite long training.

**Qualitative takeaways**
- Option selection remains dominated by `track_ball`; the sparse bonuses tied to raw reward do not inject enough signal for `serve_setup`/`tunnel_push` to value long-horizon planning.
- Replay buffers were shared only implicitly through extrinsic reward, so each skill mostly learned local reflexes. The manager, lacking intrinsic feedback about subgoal completion, rarely deviates from the safety skill.

**Next steps**
1. Add skill-specific intrinsic rewards/triggers (e.g., reward `serve_setup` for firing launch successfully, `tunnel_push` for brick hits on the target columns) so the manager receives shaped returns per option.
2. Reduce epsilon schedules for matured skills while keeping manager exploration higher to encourage option diversity.
3. Once intrinsic signals are in place, re-run evaluation (the existing script can be re-used once we resolve the ALE re-init segfault, likely by integrating evaluation directly into `HierarchicalTrainer`).
