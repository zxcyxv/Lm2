# Step-1750 standard sequential PPL protocol

## Status

- State: completed
- Authorization: user-requested
- Requested checkpoint: step 1500
- Executed checkpoint: preserved step 1750; step 1500 weights had already been
  overwritten before this evaluation was requested

## Definition

This is standard teacher-forced autoregressive perplexity, not free-running
text statistics. At every position the real prefix is encoded, `K hA` is
decoded as one next-token state, and CE is evaluated against the real next
token. All 256 next-token positions are used rather than only anchors spaced
three tokens apart.

## Data and criterion

- fixed WikiText-103 validation starts from the training run, seed 2336
- 128 windows, 256 decisions each, 32,768 total labels
- strict float32 with TF32 disabled; no inference-time noise
- test split remains unread
- the dense sequential NLL should agree within 0.05 with the logged sparse
  h1 NLL at step 1750; larger disagreement indicates an anchor-position
  sampling effect that must be reported

## Result

Evidence: [metrics.tsv](metrics.tsv) and [analysis.md](analysis.md).

- Dense standard sequential NLL: `4.626863`
- Dense standard sequential PPL: `102.19299`
- Dense next-token accuracy: `0.240723`
- Logged sparse-anchor h1 NLL: `4.665450`; dense-minus-sparse difference
  `-0.038587`, within the registered consistency tolerance

Status: `supported` that the sparse h1 metric represented standard sequential
PPL within the registered tolerance at this checkpoint. This does not recover
the unavailable step-1500 weights.
