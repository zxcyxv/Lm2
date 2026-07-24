# EXP-20260722: Query K=1, joint CE + state MSE WITHOUT stop-gradient, 13M

## Status and provenance

- State: preregistered before execution
- Authorization: user-requested variant of `EXP-20260722-query-k1-mse-ce-joint-13m`
  with the stop-gradient on the MSE target removed
- Producer: `train_query_k1_mse_ce_joint_nodetach_13m.py`
- Seed: 1337; scratch initialization; same architecture, data, schedule as
  the stop-gradient sibling
- Comparators: `EXP-20260721-query-k1-inverse-head-ablation-13m/query_inverse_rms`
  (CE only) and `EXP-20260722-query-k1-mse-ce-joint-13m` (CE + stop-gradient
  MSE, same seed/schedule)

## Question

The sibling experiment stop-gradients the MSE target (`gold_(t+1)`) so only
the `K(q)` side moves to reduce the loss. Does removing that stop-gradient —
letting gradient flow into the encoder through both the prediction and the
target computation — change one-step next-token quality or the K^2
zero-shot extrapolation behavior observed in the sibling run?

## Objective

Identical to the sibling experiment except `gold_next_states` no longer runs
under `torch.no_grad()`:

~~~text
gold_(t+1) = encoder([x_0..x_t, x_(t+1)])_(t+1)   # NOT stop-gradient; both
             sides of the MSE receive gradient this time
MSE        = mean_t || s_(t+1) - gold_(t+1) ||^2 / ||gold_(t+1)||^2
loss       = CE + MSE
~~~

## Named risk

Without stop-gradient, the MSE term can be minimized by moving either side:
`K(q)` toward `gold`, or `gold` (i.e., the shared encoder's own
representations) toward whatever `K(q)` already produces. The latter is a
representational-collapse shortcut with no counterpart in the stop-gradient
sibling: if the encoder simply reduces the diversity/scale of its own hidden
states so that the two sides become close without `K` learning a genuine
one-step transition, `gold_state_relative_mse` could look artificially good
while decoded token quality and true extrapolation do not improve. This
experiment's job is to measure whether that happens here, not to assume it
either way.

## Data and optimization

Identical to the sibling: vocabulary 8192, width 896, 2 reversible encoder
blocks, `rms-tied` head, context 256, batch 128, 1000 AdamW steps, peak LR
`3e-4`, 100-step warmup, cosine decay, gradient clip 1.0, strict FP32, TF32
disabled, seed 1337.

## Success and stopping

Same preflight gates as the sibling (finite/nonzero CE and MSE gradients into
`K`, query, and now also the embedding table; exact-inverse roundtrip
`< 1e-4`) before any optimizer step. Primary comparison is validation NLL and
accuracy against both comparators at equal steps, plus the same K^2 zero-shot
state-alignment and token-accuracy check used on the stop-gradient sibling
(`eval_k1_k2_extrapolation_check.py`). A single run, single seed: this cannot
establish whether removing stop-gradient helps or hurts in general, only in
this one matched setting.

## Scheduling note

Queued to start immediately after `EXP-20260722-query-k1-mse-ce-joint-13m`
finishes its 1000 steps on the same GPU (single-GPU machine; not run
concurrently with that job).

## Aborted first attempt (batch 128, OOM)

The first attempt at batch 128 (matching both comparators) crashed with
`torch.OutOfMemoryError` on step 2, having only completed step 1's report.
Without stop-gradient, the `gold_next_states` encoder forward pass must retain
its activations for backpropagation instead of running under `torch.no_grad()`,
roughly doubling the backward-graph memory relative to the stop-gradient
sibling at equal batch size (which peaked around 13.4 GiB at batch 128). The
registered run below uses batch 64 instead to fit in the available 15.7 GiB
GPU, keeping steps at 1000 rather than doubling to hold total tokens constant
(doubling would roughly double wall time). This is a memory-driven deviation,
not a methodological choice: this run sees half the supervised tokens of both
comparators (`ce_only` and the stop-gradient sibling), so its final numbers
are not strictly training-token-matched against them. That gap is reported
directly rather than papered over.
