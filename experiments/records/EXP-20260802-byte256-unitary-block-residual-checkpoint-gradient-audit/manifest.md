# EXP-20260802 unitary block-residual checkpoint gradient audit

## Status

- State: completed
- Source: `../EXP-20260802-byte256-complex-self-predicted-kv-unitary-block-residual-h16-ce-only-attached-stride16-13m/`
- Test split remains unread.

## Question and protocol

Which exact parameter subparts dominate the gradient after the stopped
unitary-block-residual run's step-100 and step-200 checkpoints?

For each checkpoint, restore its model and saved train-generator state, sample
the immediately following effective batch (64, microbatch 16), accumulate the
unchanged fully attached H16 CE gradient, and measure it before clipping. Log
every named parameter's L2 norm, max absolute value, and parameter-relative
norm. Also log disjoint squared-norm fractions for encoder block attention,
FFN and norms, embedding/head norm, and central input norm, Q, K, V, O and
phase tensors. Group squared sums must reproduce global squared norm to
relative error below `1e-12`.

## Evidence boundary

These are exact gradients for the reproducible next batch at each saved
post-update checkpoint. They are not the already-consumed batches that
produced the recorded step-100 and step-200 update norms. Metrics are TSV and
interpretation is Markdown; no checkpoint is written.
