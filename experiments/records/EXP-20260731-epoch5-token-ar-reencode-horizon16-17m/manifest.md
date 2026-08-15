# EXP-20260731 epoch-5 one-token hard-AR feedback audit

## Status and scope

- State: registered before this combined evaluator run
- Execution state: not run; superseded by the user's clarification that the
  requested metric was ordinary one-token teacher-forced corpus perplexity,
  not hard-token free rollout
- Status: post-hoc diagnostic requested after the continuous-vs-discrete
  latent feedback result; earlier component results make this a replication
  and consolidation rather than a blind confirmatory test
- Parent training experiment:
  `EXP-20260731-wikitext2-gpt2tok-5epoch-unified-horizon16-17m`
- Checkpoints: each model's epoch-5 `last.pt`; no parameter updates
- Seed: 1337
- Split: all 965 WikiText-2 validation chunks
- Precision: strict float32 with TF32 disabled

## Question

When the latent model is forced to commit and re-enter one hard token after
every prediction, does it retain its fixed-reference rollout advantage over
the size-matched Transformer and Grassmann baselines, and at what horizon does
any ranking crossover occur?

## Matched feedback policies

Every chunk supplies the same literal 240-token prefix and 16 held-out scoring
targets.

- `latent`: encode the current literal/generated token prefix, apply the
  state-conditioned transition once, exact-inverse decode one next-token
  distribution, append its greedy argmax token, and repeat
- `transformer` and `grassmann`: run the current literal/generated token
  prefix, append the greedy argmax token, and repeat

No path reads a held-out token inside its open rollout. Teacher logits retain
literal gold context and are reported separately.

## Metrics and criteria

For cumulative `N = 1, 2, 4, 8, 16`, report teacher and open NLL/PPL,
open-excess NLL, accuracy, and teacher/open top-1 agreement.

- H1 latent logits must reproduce the parent fixed-prefix result within normal
  float32 reduction tolerance
- report the first registered horizon at which Transformer open NLL becomes
  lower than latent hard-AR open NLL
- primary comparison: whether latent hard-AR cumulative H16 excess NLL is no
  greater than Transformer's
- retain all horizons and both baselines even when the expected crossover or
  primary comparison fails

These are fixed-reference free-rollout scores, not ordinary chain-rule corpus
perplexities. Absolute degradation with larger N is not itself interpreted as
an architectural effect.
