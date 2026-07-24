# EXP-20260722: Query K=1, Householder-orthogonal K + ActNorm, joint CE + MSE, 3000 steps, 13M

## Status and provenance

- State: preregistered before execution
- Authorization: user-requested addition of an invertible per-channel
  normalization (ActNorm) after the orthogonal `K`, run for 3000 steps
- Producer: `train_query_k1_householder_actnorm_mse_ce_joint_13m.py`
- Seed: 1337; scratch initialization
- Comparators: `EXP-20260722-query-k1-orthogonal-operator-mse-ce-joint-13m`
  (Householder `K` alone, 1000 steps: h1 acc `15.6%`, h2 hard-token `7.0%`,
  h2 direct-K^2 `1.6%`, `gold_state_relative_mse` `0.0045`),
  `EXP-20260722-query-k1-mse-ce-joint-13m` (unconstrained linear `K`,
  stop-gradient, 1000 steps: `gold_state_relative_mse` `0.0023`)

## Question

An exactly orthogonal `K` preserves `||K q|| = ||q||` exactly, so it
structurally cannot correct a magnitude mismatch between `q` and the true
future state `gold`. A quick measurement on the trained Householder
checkpoint's own validation windows: mean `||q|| = 287.7`, mean
`||gold_1|| = 291.6` (ratio `1.0135`), giving a magnitude-only relative-MSE
floor of `0.00035` -- about `8%` of the observed `gold_state_relative_mse`
(`0.0045`). So the aggregate norm mismatch is real but small; most of the
residual error is directional, not magnitude. A stronger reason to expect a
benefit: ActNorm corrects mismatch **per channel**, not just in aggregate
norm, which could address the recurring finding across every experiment
today that state cosine near `0.99+` does not translate to good top-1
accuracy (an anisotropic, channel-wise miscalibration is a plausible
mechanism neither cosine nor a pure rotation can fix).

Does adding a learned, exactly invertible per-channel affine (ActNorm) after
`K`'s output close some of this gap, on both the state-alignment metric and
h1/h2 token accuracy, relative to Householder `K` alone?

## Objective and architecture

Identical forward contract to the Householder-only sibling, with one
insertion:

~~~text
q_(t+1)      = contextual query encoder state
s_(t+1)      = K q_(t+1)                          # Householder, orthogonal
s'_(t+1)     = exp(log_scale) * s_(t+1) + bias     # ActNorm, per-channel,
                                                    # exactly invertible:
                                                    # s = (s' - bias) *
                                                    # exp(-log_scale)
y_(t+1)      = exact_inverse_decode([h_<=t, s'_(t+1)])
logits       = rms_tied_head(y_(t+1))
gold_(t+1)   = stop_gradient(encoder([x_0..x_t, x_(t+1)]))_(t+1)  # same
               online encoder, no EMA
CE           = mean cross_entropy(logits, x_(t+1))
MSE          = mean_t || s'_(t+1) - gold_(t+1) ||^2 / ||gold_(t+1)||^2
loss         = CE + MSE
~~~

`log_scale` and `bias` are both initialized to `0`, so ActNorm is exactly the
identity at init (`s' = s`) -- consistent with this project's convention that
`K` (and now everything composed with it) starts as the identity map. This
is a new, self-contained model wrapper; it does not modify the shared
`QueryK1InverseLM`/`PrefixOrbitLM`/`RevBlock` classes used by every other
experiment.

## Data and optimization

Same architecture constants as the Householder-only sibling: vocabulary
8192, width 896, 2 reversible encoder blocks, `rms-tied` head, context 256,
batch 64 (memory-driven, matching the Householder sibling's fix), peak LR
`3e-4`, 100-step warmup, cosine decay over the full run, gradient clip 1.0,
strict FP32, TF32 disabled, seed 1337. **Steps: 3000** (user-requested,
longer than every other sibling's 1000) to see whether the joint objective
keeps improving with more optimization or plateaus/regresses the way the
first stop-gradient run's h1 accuracy did between step 750 and 1000.

## Success and stopping

Preflight must confirm ActNorm is exactly the identity at init
(`log_scale=0`, `bias=0`, so `s'=s` to floating-point precision) in addition
to every gate used by the Householder-only sibling (orthogonality and
identity-at-init for `K`, exact-inverse roundtrip `< 1e-4`, finite/nonzero CE
and MSE gradients into `K`'s vectors, ActNorm's `log_scale`/`bias`, and
`query`). Primary comparison: `gold_state_relative_mse` and h1/h2 (hard-token,
direct-K^2) accuracy against the Householder-only sibling at matched steps
(1000) and at the final step (3000). A single run, single seed: this cannot
establish whether ActNorm helps in general, only in this one matched setting.
