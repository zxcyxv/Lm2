# ε-reinsertion tolerance vs training-curve extrapolation

Question: can ~30k additional training steps lower the ha-skew operator's
relative state MSE enough that direct decoding of K(h_A) yields correct
tokens (h1 extrapolation)?

## Tolerance curve (gold h_B + controlled noise, exact-inverse decode)

Checkpoint: EXP-20260723-k1-ha-skew-learned-noise-mse-ce-13m best.pt (step 6000).
256 validation windows; noise scaled so relative MSE = ε² exactly per row.

- Gold ceiling (ε=0): 99.98% — decoder self-identity is intact.
- Model's own K(h_A): relative MSE 0.00708 → 32.9% accuracy, matching the
  k-error-direction curve at the same magnitude (0.321 @ ε=0.085).
- K's residual direction is ~10x more benign than an isotropic random
  direction at equal magnitude (e.g. ε=0.085: 32% vs 3.2%): the trained
  residual partially avoids the decoder's amplifying subspace.

Accuracy thresholds in the *benign* (k-error) direction:
- 90% h1 accuracy needs relative MSE ≈ 2.5e-5
- 50% h1 accuracy needs relative MSE ≈ 4e-4
(random direction is stricter: 50% at ≈1.5e-4)

## Training-curve extrapolation (this run's own metrics.tsv)

Power-law fit of val_gold_state_relative_mse:
- global fit (steps ≥ 500): mse = 0.242 · step^(−0.386) → 4.2e-3 @ 36k steps
- optimistic local slope (steps 2250→6000, b≈0.70) → 2.0e-3 @ 36k steps

Steps needed to reach thresholds:
- 50% threshold (4e-4): ~3.7e5 steps (optimistic) to ~1.6e7 (global fit)
- 90% threshold (2.5e-5): ~1.9e7 to ~2.1e10 steps

## Verdict

36k total steps lands at 2e-3 – 4.2e-3, a factor of 5–10 above even the
50% h1 threshold and ~80–170x above the 90% threshold. 30k additional
steps cannot produce h1 extrapolation on this trajectory, and h≥2 is
strictly harder (orthogonal K is an isometry: per-step errors are carried
undamped and accumulate, so the per-step tolerance shrinks with horizon).

Theoretical caveat that makes even these step counts optimistic: K(h_A)
is a deterministic function of the prefix while h_B depends on the next
token, so E||K h_A − h_B||² is lower-bounded by E[Var(h_B | prefix)] — a
strictly positive floor whenever next-token entropy is positive. The
flattening of the fitted curve may already be the shadow of this floor;
if the floor sits above 4e-4, no number of steps crosses the threshold.
The floor is directly measurable (spread of h_B over candidate next
tokens for a fixed prefix) without any training.

Script: eval_ha_skew_epsilon_tolerance.py · Data: tolerance.tsv
