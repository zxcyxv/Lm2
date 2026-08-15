# EXP-20260803 step-10071 flow and generation audit

## Status

- State: completed; raw-carrier and preregistered ray-geometry control both
  evaluated
- Parent: 13M fused Triton rotating-frame ten-epoch run
- Fixed validation split only; test unread

## Questions and comparisons

At the three-epoch checkpoint, does the learned innovation point in the same
co-rotating direction as the causal gold-encoder trajectory? Is alignment
stronger for correctly decoded bytes, especially at shallow horizons? Also
inspect actual continuation quality using the causally correct final-prefix
anchor rather than reusing the last stride-16 training anchor.

Compare H1-reanchored generation with H16 committed blocks on identical fixed
validation prompts. This audit is descriptive and does not train or alter the
checkpoint.

## Fixed protocol

- checkpoint: epoch 3 / step 10071 from the active 13M run
- seed 1337; fixed validation starts; test unread
- 64 validation windows for velocity alignment
- actual final context position 255 is the generation/diagnostic root
- gold states are causal encoder states at positions 256 through 271
- predicted and gold innovations are both transformed by `R^-r`
- report horizon-level and correct/incorrect velocity cosine, relative error,
  norms, and decoded-byte correctness in TSV
- because the decoder reads an RMS-normalized carrier, separately report the
  corresponding unit-ray velocity and predicted-state/gold-state directional
  cosine; do not use raw radial mismatch alone to reject latent flow
- generate 128 bytes for eight prompts using H1 reanchoring and H16 blocks
- metrics and samples go to TSV; conclusions go to Markdown

## Interpretation gates

- positive cosine alone is not proof of distributional flow matching
- deterministic gold-velocity agreement is only a prerequisite for adding an
  auxiliary flow objective
- a raw-carrier failure with positive ray-velocity agreement means only the
  radial gauge failed; failure of both is evidence against an already learned
  encoder-conjugate flow
- do not call H16 generation successful from teacher-forced NLL alone
