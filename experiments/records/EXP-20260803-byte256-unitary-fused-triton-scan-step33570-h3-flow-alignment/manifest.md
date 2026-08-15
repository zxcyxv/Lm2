# Step-33570 H1-H3 flow-alignment audit

## Status

- State: superseded before metric production; user selected H1-H4 so the
  fused power-of-two backend can be evaluated directly
- Parent: completed 13M fused-scan ten-epoch run
- Split: fixed validation only; test unread

## Question and comparison

At the final epoch-10 checkpoint, do the learned H1-H3 innovations align with
the causal gold-encoder trajectory strongly enough to motivate a latent flow
matching objective?  Compare step 33570 with the already preserved step-10071
results on the identical 64 fixed starts.  Do not use H4-H16 evidence.

## Fixed protocol

- checkpoint: step 33570
- seed 1337 and the preserved 64 validation starts
- actual final prefix position 255 is the root
- horizons: H1, H2, and H3 only
- evaluation backend: exact rotating-frame cumsum for both checkpoints; the
  first preflight attempt with the fused Triton backend was rejected before
  metrics because that kernel requires a power-of-two horizon
- raw target: `g_r - R g_(r-1)`
- ray target: `normalize(g_r) - R normalize(g_(r-1))`
- transform predicted and target velocities by the same `R^-r`
- report raw/ray cosine, relative error, velocity norms, endpoint direction
  cosine, decoded-byte correctness, and correct/incorrect partitions
- compute paired sample-cluster uncertainty against step 10071 after the run
- metrics are TSV and interpretation is Markdown

## Interpretation gates

- deterministic velocity alignment is only a prerequisite for, not proof of,
  distributional flow matching
- positive cosine must be consistent across H1-H3 and non-negligible relative
  to paired sample uncertainty
- endpoint direction cosine alone is insufficient if the learned velocity is
  near zero or points elsewhere
- correctness association must survive horizon separation; an aggregate gap
  caused only by easier H1 examples is not evidence
