# Corrected process-noise effect audit

## Status

- State: completed
- Parent: completed three-arm 300-step stochastic value-write continuation
- Split: identical fixed validation only; test unread

## Question and correction

The parent metric compared a stochastic full-window causal encode with the
older prefix-only fast encode.  At noise scale zero this produced a non-zero
logit RMS, so that column mixed process-noise sensitivity with different
kernel evaluation shapes.  Preserve the parent TSV, but recompute the noise
effect by evaluating scale 0 and scale 0.05 through the identical full-window
encode, stochastic loss wrapper, explicit complex noise tape, and fused scan.

Compare the parent step-33570 checkpoint and all three step-300 arm
checkpoints.  Use seed 25618, the preserved 64 validation starts, H16, anchor
stride 16, and microbatch 16.  Report corrected deterministic/stochastic NLL,
logit RMS, top-1 disagreement, and H1--H4 ray statistics in TSV.

## Success criterion

- the deterministic arm at requested scale zero must have zero logit/NLL
  delta to numerical precision;
- stochastic arms must report only the effect of the explicit value-noise
  tape;
- do not delete or overwrite the conflicting parent metric; document its
  scope in the final interpretation.
