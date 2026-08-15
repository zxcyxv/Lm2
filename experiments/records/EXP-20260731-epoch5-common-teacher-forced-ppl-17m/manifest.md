# EXP-20260731 epoch-5 common teacher-forced PPL

## Status and scope

- State: registered before evaluation
- Authorization: the user explicitly requested actual one-token
  teacher-forced PPL rather than a block/free-rollout score
- Parent training experiment:
  `EXP-20260731-wikitext2-gpt2tok-5epoch-unified-horizon16-17m`
- Checkpoints: each model's epoch-5 `last.pt`; no validation-selected model and
  no parameter update
- Seed: 1337
- Split: all 965 WikiText-2 validation chunks
- Precision: strict float32 with TF32 disabled

## Question

Under the standard next-token teacher-forcing definition and one common token
accumulator, what is the epoch-5 validation PPL of the latent model,
size-matched Transformer, and Grassmann model?

## Evaluation

For every 256-token validation chunk, score each of the 255 targets exactly
once:

\[
\operatorname{NLL}=-\frac{1}{965\cdot255}
\sum_{b=1}^{965}\sum_{t=1}^{255}
\log p_\theta(x_{b,t+1}\mid x_{b,1:t}).
\]

- latent: causally encode the literal 255-token context, apply its registered
  state-conditioned transition once at every anchor, and exact-inverse decode
  the corresponding next-token distribution
- Transformer and Grassmann: ordinary causal forward pass over the same
  literal 255-token context
- all models use the identical targets, float32 CE implementation, reduction,
  and token denominator
- no argmax, generated token, open rollout, block reset, or horizon weighting
  enters this metric

## Success criteria

- exactly `965 * 255 = 246,075` target tokens are scored per model
- all NLL values are finite
- latent validation PPL is lower than the size-matched Transformer's
- report all three results even if the ranking conflicts with the hypothesis

