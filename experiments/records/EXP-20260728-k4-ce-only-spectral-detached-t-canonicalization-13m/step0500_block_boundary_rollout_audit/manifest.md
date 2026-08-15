# Step-500 matched block-boundary AR versus parallel rollout audit

## Status

- State: preregistered before running this producer
- Authorization: user requested a fair comparison after identifying that the
  earlier AR reference received gold-token teacher forcing every token
- Parent checkpoint: `step0500.pt`
- Expected checkpoint SHA-256:
  `b7b679d98ef0ca4b32d8a33578ecc5e8e76cd889ca4706ff3510d96666956742`
- Validation-start SHA-256:
  `2cbb42d154272639d91928f262187fd6d3bcf52bf294c728b6b884e0bc7cb025`
- Split: validation only; test remains unread
- Seed and preserved validation starts: parent seed 1337 and all 128 parent
  starts

## Question

When both policies receive gold context only at the same four-token block
boundaries, does the block-frozen parallel spectral orbit assign similar
held-out-token likelihood to a sequential AR rollout that feeds back its own
greedy tokens inside each block?

The earlier audit compared

`p_AR(x_(t+j) | gold x_<=t+j-1)`

against

`p_parallel(x_(t+j) | gold x_<=t)`.

That remains a valid measurement of the likelihood cost of removing
within-block gold conditioning, but it is not a matched generation-parity
comparison. This audit does not overwrite that evidence.

## Matched policies

At anchors 0, 4, ..., 252, both policies start from the identical literal
gold prefix ending at the anchor.

1. `boundary_ar_rollout`
   - compile a fresh prefix-conditioned spectral K for one token;
   - greedily select that token;
   - append and canonically re-encode the selected token;
   - repeat until four tokens have been selected;
   - discard the generated branch at the end of the block and start the next
     independent branch from its preserved gold boundary.
2. `parallel_block4`
   - compile K once from the same boundary state;
   - compute `K hA`, ..., `K^4 hA` in the registered closed form;
   - jointly decode and greedily select four tokens;
   - do not consume corpus or selected tokens inside the block.

Both paths are target-independent after the shared boundary. Corpus
continuations are read only after both logits and greedy actions exist, for
scoring.

## Fixed evaluation

- WikiText-103 validation split
- 128 preserved sequences
- context 256 and four-token continuation
- 64 block anchors per sequence; 32,768 scored tokens
- greedy action selection
- strict float32; TF32 disabled
- CUDA microbatch 4
- no T-corrector in either generation path

## Metrics

- held-out target NLL and `exp(mean NLL)` for both policies
- parallel-minus-AR rollout NLL and PPL ratio
- symmetric PPL ratio `max(PPL_AR,PPL_parallel)/min(...)`
- horizon-1 through horizon-4 NLL, PPL, accuracy, and policy token agreement
- overall token agreement and four-token exact-block agreement
- horizon-1 maximum logit error as a structural same-start check

Because the AR contexts contain its own greedy tokens rather than the corpus
tokens, `exp(mean NLL)` is a rollout PPL-like diagnostic, not standard corpus
perplexity. It must remain labeled separately from the earlier
token-teacher-forced AR PPL.

## Registered interpretation

The primary likelihoods are considered similar if

`abs(parallel_nll - boundary_ar_nll) <= log(1.10)`,

equivalently if the symmetric PPL ratio is at most 1.10.

Generation parity is stronger and is supported only if:

- horizon-1 maximum logit error is at most `1e-5`;
- overall greedy token agreement is at least 0.90; and
- exact four-token block agreement is at least 0.75.

Likelihood similarity without token parity means the two policies have
similar average held-out loss but are not the same greedy generator. Numeric
metrics are written to TSV and interpretation is written separately to
Markdown.
