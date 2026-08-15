# Update-direction alignment audit

## Status

- State: completed; exact native update-1001 direction reproduced
- Parent audit: `manifest.md`
- Source: native no-QKV-pre-norm `step1000.pt`
- Test split remains unmaterialized and unread.

## Question

The registered update 1001 increased its own exact training-batch CE. Is the
actual clipped-AdamW parameter delta already a first-order ascent direction
for the current raw gradient, or is it first-order descent whose finite update
is reversed by curvature/nonlinearity?

## Fixed comparison

- Restore the native step-1000 model, fused AdamW state, and training-data
  generator.
- Draw exactly the same next train batch as the parent audit: seed/state from
  the checkpoint, batch 64, microbatch 16, context 256, H16, stride 16.
- Reuse the parent producer's common exact-batch gradient routine and existing
  parameter-group definition.
- Reproduce only the already registered update 1001 in memory. Do not perform
  a second optimizer update and do not continue training.
- For the global vector and every existing parameter group, measure the raw
  current gradient `g`, actual parameter delta `delta = theta_1001-theta_1000`,
  `g dot delta`, their cosine, and the equivalent cosine between `g` and
  `-delta`.
- Interpret `g dot delta` as the first-order predicted CE change. A negative
  value is first-order descent; a positive value is first-order ascent.

## Outputs and success criteria

- Metrics: `update_direction_alignment.tsv`
- Interpretation: `update_direction_interpretation.md`

The audit succeeds only if the reproduced pre/post same-batch CE, raw gradient
norm, and parameter deltas agree with the parent audit within numerical
tolerance; the disjoint group dot products must sum to the global dot product.
All conflicting signs are preserved rather than resolved by deleting evidence.

## Evidence boundary

This is one local first-order decomposition on the exact native update-1001
batch. It distinguishes optimizer-direction misalignment from a finite-step
loss reversal at this point only. It does not identify the Hessian terms,
reconstruct the historical step-900 spike, or establish long-run behavior.

## Producer

```bash
python eval_byte256_complex_self_predicted_kv_no_qkv_prenorm_step1000_update_direction.py
```
