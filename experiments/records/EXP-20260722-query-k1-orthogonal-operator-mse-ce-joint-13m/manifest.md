# EXP-20260722: Query K=1, orthogonal operator, joint CE + stop-gradient state MSE, 13M

## Status and provenance

- State: preregistered before execution
- Authorization: user-requested orthogonal/spectrum-1 reparameterization of
  `K`, motivated by: (1) the K=3 audit finding a near-singular direction
  (minimum singular value `0.006`) plausibly linked to repetition collapse,
  and (2) the concrete argument that a contracting `K` cannot support
  parallel generation, since the hidden state would vanish over future
  horizons under repeated application
- Producer: `train_query_k1_orthogonal_mse_ce_joint_13m.py`
- Seed: 1337; scratch initialization
- Comparators: `EXP-20260721-query-k1-inverse-head-ablation-13m/query_inverse_rms`
  (CE only, linear `K`), `EXP-20260722-query-k1-mse-ce-joint-13m` (CE +
  stop-gradient state MSE, linear `K` -- the best K^2 extrapolation result
  so far: h2 direct-K^2 state cosine `0.998`, accuracy `3.9%`)

## Question

Does constraining `K` to be exactly orthogonal (all singular values `= 1`,
so repeated application `K, K^2, K^3, ...` neither shrinks nor grows the
hidden state's norm) improve K^2 zero-shot extrapolation quality relative to
the unconstrained linear `K` used by every prior experiment in this project,
under the same joint CE + stop-gradient state-MSE objective that already gave
the best extrapolation result?

## Parameterization

Reviewed against the normalizing-flows survey
(`docs/1908.09257v4.pdf`, Kobyzev et al.), section 3.2.3 (orthogonal linear
flows via Householder transforms, Tomczak and Welling 2016). Two ways to
force spectral norm exactly `1` were considered: a Householder-reflection
product (matches the user's explicit request), and a matrix exponential of a
skew-symmetric generator (`K = expm(A - A^T)`, also exactly orthogonal and
trivially `K = I` at `A = 0`). **Householder was used**, per direct
instruction, with a specific trick to still start at identity:

~~~text
K            = H_1 H_2 ... H_m,  H_i = I - 2 v_i v_i^T / ||v_i||^2
m            = 32 reflection vectors, each in R^896
init         : v_(2i) = v_(2i+1) for every pair -- a reflection composed
               with itself is the identity (H H = I) for ANY vector, so
               K = I exactly at init, matching this project's established
               convention -- while the two vectors in each pair remain
               independent learnable parameters free to diverge under
               training
s_(t+1)      = K q_(t+1)          # same forward contract as every prior
                                   # K=1 query-based experiment
~~~

A single Householder reflection alone is never close to identity (normalizing
the vector erases any information about its scale, so even a tiny-norm
random vector gives a reflection just as far from identity as any other
direction) -- the paired-vector trick above is what makes identity
initialization possible while still literally composing reflections.
Composition of any number of reflections is exactly orthogonal for any
parameter values, so orthogonality does not need to be preserved by
training -- it is a structural invariant, checked numerically each
evaluation only as a floating-point sanity gate.

## Objective

Identical to the stop-gradient sibling
(`EXP-20260722-query-k1-mse-ce-joint-13m`): `loss = CE + relative_MSE(K(q),
stop_gradient(gold_(t+1)))`, gold from the same online encoder. Only `K`'s
parameterization changes (orthogonal instead of unconstrained linear).

## Data and optimization

Identical to all K=1 13M siblings: vocabulary 8192, width 896, 2 reversible
encoder blocks, `rms-tied` head, context 256, batch 128, 1000 AdamW steps,
peak LR `3e-4`, 100-step warmup, cosine decay, gradient clip 1.0, strict
FP32, TF32 disabled, seed 1337.

## Success and stopping

Preflight must additionally confirm `K` is orthogonal at init (`K^T K = I` to
within numerical tolerance) and that `A`'s gradient is finite and nonzero, in
addition to the gates used by the linear-`K` siblings (exact-inverse
roundtrip `< 1e-4`, finite/nonzero CE and MSE gradients). Primary comparison:
h1 accuracy, h2 hard-token and direct-K^2 accuracy/state-cosine
(`eval_k1_h2_three_way_check.py`-style) against the linear-`K` stop-gradient
sibling. A single run, single seed: this cannot establish whether an
orthogonal `K` is better in general, only in this one matched setting.

## Scheduling note

Queued to start after `EXP-20260722-query-k1-mse-ce-joint-ema-rampup-13m`
finishes on the same single GPU. That run was manually stopped early per
user instruction before this one started.

## Aborted first attempt (batch 128, OOM)

Batch 128 (matching every other sibling) crashed with
`torch.OutOfMemoryError` immediately at step 1's optimizer step. The 32
sequential Householder reflections each require their intermediate output
retained for backpropagation (since each step's backward depends on the
previous step's output), adding roughly `32 x batch x context x width x 4
bytes ~= 3.6 GiB` of activation memory beyond the unconstrained-linear-`K`
siblings, which pushed total usage past the 15.7 GiB GPU budget. The
registered run below uses batch 64 instead; like the earlier no-stopgrad
OOM fix, this is a memory-driven deviation, not a methodological one, and
means this run sees half the supervised tokens per step of the linear-`K`
comparators at equal step count.
