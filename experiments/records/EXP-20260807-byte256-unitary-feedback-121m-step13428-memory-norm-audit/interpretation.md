# 121M current-checkpoint memory norm audit

## Scope

- Source checkpoint: `outputs/experiments/EXP-20260807-byte256-unitary-feedback-h4-width4352-121m-lr2e-4/last.pt` (step 13428)
- Fixed validation only: 4 windows x 16 roots = 64 trajectories, rolled through H16
- This is a forward-state audit. It does not infer causality for loss or gradient behavior.

## Direct result

The raw memory Frobenius norm increased from mean `21027.309174` at H1 to `186328.768618` at H16 (`8.861x`). At H16, individual trajectory norms ranged from `30296.167850` to `355715.225970`.

Across H2--H16, the cross term was negative in `0.00%` of trajectory-steps, while total memory energy actually decreased in `0.00%`. Thus writes do sometimes cancel the rotated state; whether that cancellation is strong enough to overcome the write's own positive energy is a separate question.

## Exact energy decomposition

Summed over all measured trajectories and horizons:

- total `Delta ||S||^2`: `2.839769e+12`
- total `||kv^dagger||^2`: `2.399105e+11`
- total `2 Re<US,kv^dagger>`: `2.599859e+12`
- decomposition closure error: `1.066390e+04` (relative `3.755e-09`)

The first term is always non-negative. Negative/complex entries in `kv^dagger` affect the signed cross term, not the positivity of `||kv^dagger||^2`.

## Coherence versus an incoherent-write baseline

At H16, `||S|| / sqrt(sum_r ||W_r||^2)` had mean `3.376977`, median `3.400470`, and range `[2.806558, 3.546877]`. A value near one is the energy scale of mutually incoherent writes; values above one indicate net constructive cross-term accumulation.

## About negative KV entries

The real part of actual write entries was negative on average in `50.34%` of entries. This does not by itself imply Frobenius-norm cancellation: a complex matrix has no global positive/negative ordering, and norm growth is decided by the inner product with the already-rotated memory.

See `metrics.tsv` for per-horizon distributions, `samples.tsv` for every trajectory-step, and `head_metrics.tsv` for channel-level decomposition.
