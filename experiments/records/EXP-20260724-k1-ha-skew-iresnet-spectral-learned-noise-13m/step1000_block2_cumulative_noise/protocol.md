# Step-1000 block-2 cumulative-noise evaluation protocol

## Status

- State: completed
- Authorization: user-requested
- Checkpoint: fixed step 1000

## Question and comparison

Does using the learned training-time noise at inference change block-2
generation relative to clean block-2 decoding?

- Clean: `z1 = K hA`, `z2 = K^2 hA`
- Cumulative-noise: `z1 = K hA + eps1`,
  `z2 = K z1 + eps2 = K^2 hA + K eps1 + eps2`

The noise predictor and latent RMS scaling are exactly those used during
training. No parameter is updated.

## Data, seeds, and evaluation

- WikiText-103 validation split only; test remains unread
- the same five fixed 64-token prompts as the clean step-1000 audit
- 64 greedy continuation tokens, with real token re-grounding every two tokens
- four stochastic draws, seeds 4367 through 4370
- comparisons: distinct-1, distinct-2, immediate repetition, exact generated
  text, learned log-sigma, relative noise norm, and numerical residual of the
  expanded `K^2 hA + K eps1 + eps2` identity

## Success criterion

This is a mechanism check, not a model-selection run. The cumulative-noise
variant counts as a material behavioral change only if at least one aggregate
diversity/repetition metric changes by 0.01 absolute from clean block-2 in the
mean over the four fixed draws. Text quality remains a qualitative observation
and is not promoted to a supported claim from five prompts.

## Result

Evidence: [metrics.tsv](metrics.tsv), [diagnostics.tsv](diagnostics.tsv), and
[generation.md](generation.md).

- Clean block-2: distinct-1 0.3125, distinct-2 0.615873, immediate repetition
  0.193651.
- Four-draw cumulative-noise mean: distinct-1 0.3125, distinct-2 0.603175,
  immediate repetition 0.203175.
- Mean relative noise norm was approximately 0.00263. The expanded recurrence
  identity agreed to within `5.72e-6` maximum absolute error.

Status: `inconclusive` for general text quality; `supported` within these
fixed draws that inference-time cumulative noise caused a material diversity
change under the registered threshold, but it did not improve the aggregate
metrics consistently.
