# Grassmann epoch 12 versus width-336 latent epoch 9

## Status

- State: preregistered before generation
- Authorization: user requested a quick comparison despite unmatched epochs
- Grassmann checkpoint: validation-selected best at epoch 12, PPL `238.57`
- Latent checkpoint: width 336, depth 2, epoch 9, target-free h1 mixture PPL
  `284.55`

## Fixed comparison

- WikiText-2 validation split
- the same eight seed-1337 chunks used by the epoch-2 comparison
- 64 prompt tokens and 64 generated tokens
- Grassmann: greedy autoregressive generation
- latent model: target-free prior-argmax block 4
- innovation noise seed 3361
- greedy token choice for both

## Evidence boundary

Epoch and validation PPL are unmatched, so this comparison cannot isolate
architecture. It records sentence quality and failure modes only. Metrics are
TSV and decoded evidence/interpretation are Markdown.

## Execution

- State: completed
- Eight prompts generated without CUDA or non-finite failures
- Raw decoded continuations: `generation.md`
- Aggregate and per-sample metrics: `metrics.tsv`, `sample_metrics.tsv`
- Interpretation: `interpretation.md`
