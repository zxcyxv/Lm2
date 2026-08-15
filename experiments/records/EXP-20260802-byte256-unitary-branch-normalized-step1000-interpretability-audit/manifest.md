# EXP-20260802 step-1000 interpretability audit

## Status

- State: completed
- Source checkpoint: step 1000 of the exact step-300 continuation
- Validation split only; test unread

## Question

Without imposing `R=I`, does the learned H16 latent trajectory expose a
non-degenerate and grounded sequence of measurements on real byte contexts?

## Protocol

- Use the first eight fixed validation windows and the last stride-16 anchor
  (position 240), leaving sixteen known future labels for inspection.
- Print the visible byte context suffix, gold H16 bytes, and stepwise decoded
  top-1 bytes/probabilities.
- Decompose every measurement correction exactly by originating write:
  `delta_r = sum_(n<=r) delta_(r,n)`. Use signed projection onto total delta as
  the relative interference map and report reconstruction error and effective
  rank.
- Because learned `R` is retained, transport every correction into the final
  H16 frame and verify
  `z_H = R^H z_0 + sum_r R^(H-1-r) delta_r`.
- Report correction decay, memory Frobenius growth, interference-map effective
  rank, prediction accuracy, and collapse/repetition indicators.

## Evidence boundary

Decoded bytes ground each latent step only through the shared decoder. They do
not prove human-like reasoning or assign a natural-language proposition to a
write. Signed projection is an exact additive correction decomposition at a
fixed forward trajectory, not an end-to-end causal or gradient attribution.
No model state is modified.
