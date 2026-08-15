# Step-500 raw versus post-hoc collapse-projection audit

## Status

- State: preregistered before running this producer
- Authorization: user requested same-checkpoint quality evaluation before and
  after collapse projection
- Parent checkpoint: `step0500.pt`
- Expected checkpoint SHA-256:
  `b7b679d98ef0ca4b32d8a33578ecc5e8e76cd889ca4706ff3510d96666956742`
- Validation-start SHA-256:
  `2cbb42d154272639d91928f262187fd6d3bcf52bf294c728b6b884e0bc7cb025`
- Split: validation only; test remains unread

## Question

For the CE-only model trained with the raw alpha-zero block orbit, what
happens to four-token parallel likelihood when the same frozen checkpoint is
evaluated with the calibrated hard amplitude-shell projection enabled
post hoc?

This is an inference ablation, not a matched-training causal estimate. No
model is retrained and no parameter is updated.

## Paired paths

Both paths use the same checkpoint, literal gold block boundaries, encoded
prefixes, validation starts, and corpus targets.

1. `raw_parallel`: registered training path with `alpha=0`,
   `q_j = K_A^j h_A`.
2. `posthoc_projected_parallel`: set the existing alpha buffer to one without
   changing any weight, then use the checkpoint's preserved calibrated shell
   in the registered closed-form projection path.

The T corrector is unused in both paths. The shell was calibrated before the
parent run but excluded from its optimizers and had no effect on alpha-zero
training.

## Fixed evaluation

- WikiText-103 validation split
- parent seed 1337 and all 128 preserved validation starts
- context 256
- anchors 0, 4, ..., 252
- 8,192 blocks and 32,768 target tokens
- four jointly decoded horizons
- strict float32; TF32 disabled
- CUDA microbatch 4

## Metrics and interpretation

- overall and horizon-specific NLL, `exp(mean NLL)`, and top-1 accuracy
- projected-minus-raw NLL and projected/raw PPL ratio
- top-1 token agreement and exact four-token block agreement
- shell-radius error and alpha-zero raw nesting error as structural checks

Classification is fixed before measurement:

- `projection_improves` if projected-minus-raw NLL is at most `-0.05`;
- `approximately_neutral` if its absolute value is below `0.05`;
- `projection_degrades` if it is at least `+0.05`.

A post-hoc improvement supports using the projection at inference for this
checkpoint, but does not establish what jointly training with projection
would do. Numeric metrics are TSV and interpretation is Markdown.
