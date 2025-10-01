# Detector-Augmented Object Encoder Plan

## Goal
Introduce an auxiliary detector that extracts structured Breakout entities (paddle, ball, brick layout) each frame and feed those features to downstream RL policies alongside the standard pixel stack.

## Variants
1. **RAM-Tap Detector (Priority)** — read Atari RAM addresses for precise entity state. Serves as the initial implementation and performance baseline.
2. **Pixel/Color-Threshold Detector** — operate solely on rendered frames to evaluate portability beyond Atari RAM introspection.

## Immediate Tasks
- [ ] Catalogue RAM offsets for paddle position, ball position/velocity, and brick grid; record sources in `docs/references.md`.
- [ ] Implement `BreakoutRamDetector` producing normalized feature vectors and basic diagnostics (valid flags, value ranges).
- [ ] Add integration hook in `src/simple_game/train.py` (or a new wrapper) to concatenate detector features with CNN embeddings.
- [ ] Design corresponding policy class `DetectorAugmentedDuelingCnnPolicy` with configurable feature fusion (concat vs MLP).
- [ ] Update evaluation tooling to log detector outputs for debugging (e.g., mismatch counts).

## Follow-Up Tasks (Pixel Detector)
- [ ] Prototype color-threshold tracker and benchmark detection accuracy on recorded videos.
- [ ] Abstract detector interface so policies can swap between RAM and pixel sources without code changes.
- [ ] Compare performance/sample efficiency across detector variants and raw-pixel baselines; log results in `experiments.md`.

## Open Questions
- How many detector features are sufficient (do we need brick-by-brick status or aggregate counts)?
- Should detector outputs replace the pixel input entirely or augment it? Initial plan: concatenate to preserve visual cues.
- Do we need an auxiliary loss (e.g., reconstruction) to stabilize combined representations?

## Milestones
1. RAM detector wired through training/evaluation with new config `configs/objcentric_breakout_ram.yaml`.
2. Pixel detector achieving ≥95% detection accuracy on validation frames and integrated into training.
3. Comparative study report summarizing reward curves, serve success rates, and compute overhead for each variant.
