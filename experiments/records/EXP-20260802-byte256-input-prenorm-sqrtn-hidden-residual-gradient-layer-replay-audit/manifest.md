# EXP-20260802 exact gradient-layer replay audit

## Status

- State: registered; exact replay pending
- Source producer: `../EXP-20260802-byte256-complex-self-predicted-kv-input-prenorm-sqrtn-hidden-residual-h16-ce-only-attached-stride16-13m/`
- Test split remains unread.

## Question

Which disjoint parameter layer accounts for the raw global gradient norm and
its step-100/step-300 events in the input-pre-norm, sqrt(count), raw hidden
residual H16 producer?

## Exact replay protocol

- Recreate the source model from seed 1337 and run the inherited preflight.
- Replay the same train split, batch 64, microbatch 16, H16 objective, AdamW,
  LR schedule, four-way accumulation, and clip-then-update order through step
  300.
- At steps 1, 50, 100, 200, and 300, capture every named parameter gradient
  after all microbatches and before clipping.
- Report per-parameter L2/max-absolute gradient and disjoint group L2. Group
  squared fractions must sum to the global squared norm.
- Compare the replay global norm against the source `metrics.tsv` value at
  every target step. Checkpoint-next-batch gradients are not substituted for
  the recorded update gradients.

Groups separate tied embedding/head, encoder blocks 0 and 1, encoder head
norm, and every central input norm/Q/K/V/O/phase parameter tensor.

## Success and evidence

Replay succeeds only if all gradients are finite, every target row exists,
the disjoint group squared sum matches the global squared norm within `1e-12`
relative error, and replay/source global norms agree within `1e-4` absolute.
The dominant layer is determined by squared-norm fraction, not raw tensor
size or an unmatched validation batch.

Metrics are TSV and interpretation is Markdown. No checkpoint is produced.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python eval_byte256_input_prenorm_sqrtn_hidden_residual_gradient_layers.py
```

## Evidence boundary

This attributes parameter-gradient magnitude for the exact training updates.
It does not by itself decompose a dominant parameter tensor by individual
horizon/use site; that follow-up is required only if one shared tensor
dominates the layer attribution.
