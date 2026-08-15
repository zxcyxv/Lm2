# EXP-20260807 121M current-checkpoint memory norm audit

## Status

- State: scheduled
- Source: `../EXP-20260807-byte256-unitary-feedback-h4-width4352-121m-lr2e-4/`
- Checkpoint: read-only `last.pt`, expected saved step 13,428
- Validation split only; test unread

## Question

On actual state-dependent recurrence trajectories, does the raw complex memory
`S` grow through H16? If it grows, distinguish the necessarily non-negative
self-energy of each `W = kv^dagger` write from constructive or destructive
alignment with the rotated existing memory.

## Protocol

- Restore the 121,336,064-parameter checkpoint without optimizer updates.
- Use the first four already-registered fixed validation windows.
- Encode all 16 stride-16 prefix roots and roll each through H16, for 64
  trajectories and 1,024 trajectory-steps.
- At each step measure `||S||`, `||US||`, `||W||`, `||US+W||`, Q/K/V norms,
  hidden/read norms, and the exact decomposition
  `Delta ||S||^2 = ||W||^2 + 2 Re <US,W>`.
- Record trajectory-level values, per-horizon distributions, and per-head
  distributions. Float64 reductions audit float32/complex64 forward tensors.

## Interpretation criteria

- Norm growth is an empirical fact for this checkpoint only if the measured
  H16 carrier is larger than H1 across the fixed trajectories.
- A negative cross term is cancellation. Total memory energy decreases only
  when cancellation exceeds the write's own positive energy.
- The real-part sign distribution of complex writes is descriptive only;
  complex values do not possess a scalar positive/negative ordering.
- `||S_H|| / sqrt(sum_r ||W_r||^2)` compares actual accumulation with an
  incoherent-write energy baseline; it does not by itself measure task value.

## Producer

```bash
python eval_byte256_unitary_feedback_121m_memory_norm_audit.py \
  --checkpoint outputs/experiments/EXP-20260807-byte256-unitary-feedback-h4-width4352-121m-lr2e-4/last.pt \
  --examples 4 --microbatch 1 --horizons 16 --anchor-stride 16 --device cuda
```
