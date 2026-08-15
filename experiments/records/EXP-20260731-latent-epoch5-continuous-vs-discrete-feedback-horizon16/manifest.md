# EXP-20260731 latent continuous-vs-discrete feedback audit

## Status and scope

- State: registered before this ablation was executed
- Status: post-hoc diagnostic prompted by the unified epoch-5 horizon result;
  it is not a preregistered success criterion of the parent three-model
  experiment
- Parent experiment:
  `EXP-20260731-wikitext2-gpt2tok-5epoch-unified-horizon16-17m`
- Checkpoint: the parent's latent epoch-5 `last.pt`, without parameter updates
- Seed: 1337
- Split: WikiText-2 validation only
- Expected chunks: 965
- Tokenization and chunking: the parent's GPT-2 tokenizer and fixed
  non-overlapping 256-token chunks
- Precision: strict float32 with TF32 disabled

## Question

Holding the trained latent model, literal prefix, first prediction, targets,
and CE accumulator fixed, does continuous latent-state feedback accumulate
less fixed-reference rollout NLL than hard-token feedback followed by literal
encoder re-entry?

## Matched paths

Every validation chunk uses tokens 1--240 as a literal prefix and tokens
241--256 only as scoring targets.

1. `continuous_state`
   - encode the literal prefix once
   - recursively apply the state-conditioned transition for 16 steps
   - causally exact-inverse decode the resulting future-state tape
   - do not feed a selected token back inside the block
2. `discrete_argmax_reencode`
   - encode the same current literal/generated prefix
   - apply the same state-conditioned transition once and exact-inverse decode
     the next-token logits
   - select `argmax`, append that token, and rerun the literal encoder
   - repeat for 16 steps

Neither open path may read a held-out suffix token. Both are scored against
the same suffix only after producing logits. The first-step logits must match;
feedback can affect only horizons 2--16.

## Metrics

For `N = 1, 2, 4, 8, 16`, report the cumulative:

- open token NLL and PPL against the fixed gold suffix
- teacher token NLL and PPL
- `open_excess_nll = open_nll - teacher_nll`
- open accuracy and teacher/open top-1 agreement

Also report first-step maximum absolute logit difference and top-1 agreement
between the two paths. These are fixed-prefix rollout scores, not ordinary
chain-rule corpus perplexities.

## Success criteria

- first-step top-1 agreement is exactly 1.0 and maximum absolute logit
  difference is at most `1e-5`
- cumulative N=16 `open_excess_nll` for `continuous_state` is lower than for
  `discrete_argmax_reencode`
- retain and report all registered horizons even if the result conflicts with
  the hypothesis

