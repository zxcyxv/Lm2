# Step-1000 one-update parameter-group swap

## Status

- State: completed
- Parent audit: `manifest.md`
- Source checkpoint: ignored native `step1000.pt` from the no-QKV-pre-norm
  boundary-postnorm H16 run
- Test and validation splits remain unread; this audit uses only the exact next
  training batch stored by the source checkpoint's generator state.

## Question

After reproducing native optimizer update 1001, which isolated parameter-group
delta is sufficient to change the same-batch CE and the raw H1/H16 recurrent
scales? This is a one-update local causal counterfactual, not a claim about
long-run training stability.

## Fixed comparison

- Seed and data order: source seed 1337 and the exact checkpointed generator
  state.
- Split: WikiText-103 raw-byte train only.
- Batch: 64 windows of length 272, split into four physical microbatches of 16.
- Objective: the source mean CE over 16 anchors and 16 horizons.
- Optimizer: the checkpointed fused AdamW state, source 6000-step schedule,
  update index 1000, and clip norm 1.0.
- `pre_update`: all step-1000 parameters.
- `full_post_update`: every trainable parameter after the reproduced update.
- Atomic hybrids: start from `pre_update` and copy only one post-update delta
  for each encoder block norm/attention/FFN submodule, central Q/K/V/O, or the
  two phase tensors.
- Aggregate hybrids: encoder block 0, encoder block 1, all encoder parameters,
  all central projections, all phases, and all central parameters.

Every hybrid is evaluated on the same already-drawn batch. No hybrid receives
another optimizer update. Frozen embedding and head buffers remain identical.

## Metrics and success criteria

`update_group_counterfactual.tsv` records parameter count and delta norm,
same-batch total/H1/H16 CE, encoded-root scale, and H1/H16 P, innovation W,
stored Z, decoded-hidden, and logit scales, together with deltas from the
pre-update model. `group_swap_members.tsv` records the exact parameter names.
Interpretation is written separately to `group_swap_interpretation.md`.

The audit succeeds only if:

- the source model, optimizer, and generator all load at step 1000;
- the full update is finite and uses exactly the source learning rate;
- atomic groups form an exact, non-overlapping partition of all trainable
  parameters;
- the union of all atomic post-step deltas exactly reproduces the full
  post-update state and forward metrics;
- every counterfactual and any conflicting direction is retained.

Pre/post model states are stored only in the ignored output tree. No batch,
checkpoint, or smoke artifact is added to Git.

## Producer

```bash
python eval_byte256_complex_self_predicted_kv_no_qkv_prenorm_step1000_group_swap.py
```
