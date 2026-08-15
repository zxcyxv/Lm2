# EXP-20260815 time-varying scan, contracting memory + linear read, 13M

## Status

- State: running (launched 2026-08-15)
- Producer: `train_time_varying_scan.py --memory-decay --no-read-norm`
- Record: `outputs/decay_linearread_13m_6k/`

## Question

The forcing-tape memory scan transports `S` on the unit circle
(`S_j = U S_(j-1) + W_j`), so `S` is a lossless accumulator and the read is
divided by `||S||_F` to stay bounded. That division is not what any linear
attention model does: the field bounds the state on the input side (RetNet
fixed decay, GLA/Mamba-2 gating, DeltaNet contraction) and normalizes after
the read, never by the state's own matrix norm.

Does replacing the division with a contraction change convergence at 13M?

## Change under test

Two edits, applied together as one arm:

- `S_j = lambda * U S_(j-1) + W_j`, `lambda = 0.999 * sigmoid(logit)` learned
  per `(head, key)`, initialised `0.95`.
- the `||S||_F` division is dropped; the read stays linear.

They are one arm because neither is sound alone: the division is what
currently bounds the read, and a contraction is what makes dropping it
bounded. With the QKV prenorm giving `||W_j|| <= B`, the state obeys
`||S|| <= B/(1-lambda)` uniformly in the horizon count.

## Comparison

Baseline `outputs/nobottleneck_13m_6k/` (completed, best block NLL 2.9190).
Identical in every other respect and launched from the same defaults:

- width 1264, heads 8, key_dim 41, value_dim 79, shared_kv_heads
- horizons 16, anchor_stride 16, context 256
- peak_lr 3e-4, warmup 100, schedule 6000, steps 6000, clip_norm 1.0
- batch 64 (microbatch 16 x4), seed 1337, scan backend eager
- byte-level `data/wikitext103_bytes`, train split; validation split for
  metrics; test unread

Parameter counts differ by exactly `heads * key_dim = 328` (the new decay
parameter): 13,323,520 -> 13,323,848.

## Note on the objective

This trainer applies cross entropy to **all 16 horizons with equal weight**
(`F.cross_entropy` over the flattened `[batch, anchors, horizons]` logits).
It has no H4-train / H16-monitor split. `val_block_nll` is the mean of the 16
per-horizon NLLs, i.e. the same quantity as `train_ce` on the other split.
The H5--H16 monitor framing in `README.md` belongs to the earlier
`byte256-unitary-feedback` lineage and does not apply here, so block NLL from
the two lineages must not be compared directly.

## Interpretation criteria

Recorded before the run; the operator decides acceptance.

- Primary: monotonicity of `train_ce` and `val_block_nll`. The registered
  concern is loss that descends and then rebounds, not the level of any
  out-of-objective horizon.
- Secondary: `gradient_norm` against `clip_norm = 1.0` (how long the run stays
  clipped) and `phase_gradient_norm^2 / gradient_norm^2`.
- The learned `lambda` distribution at the end, against its 0.95 start.

Known risk, observed in the launch smoke test: dropping the division raises
the step-1 gradient norm to ~625 against the baseline's ~129, because the
read is no longer rescaled to O(1). The output projection init and a possible
post-read normalisation were left untouched for this arm.

## Verification of the implementation

- full test suite passes (123 tests)
- defaults bit-identical: read states differ by 0.000e+00 with the new
  options off
- the parallel scan with `|multiplier| = 0.9` matches a serial loop to
  9.8e-07 (float32)
- `||S_H||` saturates in H: 7.58 (H=1), 61.30 (H=16), 73.41 (H=64),
  73.42 (H=256), against the geometric bound `1/(1-lambda) = 10`
- horizon 1 is unaffected by decay (`S_0 = 0`), so the run's
  `h1_scan_feedback_error` is still 0.000e+00
