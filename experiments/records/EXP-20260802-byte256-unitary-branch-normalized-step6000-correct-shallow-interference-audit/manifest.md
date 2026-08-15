# EXP-20260802 step-6000 correct shallow-interference audit

## Status

- State: completed
- Source artifacts: step-6000 phase/interference and interpretability audits
- Validation split only; test remains unread

## Question

Within H1--H4, where the model already predicts useful bytes, do exactly
correct predictions exhibit the intended selective constructive/destructive
interference more clearly than incorrect predictions? Correct spaces and
correct non-space bytes are reported separately so corpus-frequency success is
not mistaken for semantic selection.

## Fixed protocol

- same eight fixed validation contexts from seed 1337 + 999
- restrict to H1--H4: 32 token predictions total
- join exact correctness and visible target/prediction bytes to the exact
  channel/write decomposition
- compare correct, incorrect, correct-space, and correct-nonspace groups on
  phase coherence, positive/negative mass, cancellation ratio, effective
  channel-write count, dominant-term share, current-write share, and
  absolute-projection-weighted contribution age
- preserve every token-level row in TSV; this is a successful-case mechanism
  case study, not a population-level causal test

## Indicative success pattern

Correct non-space predictions should show some combination of higher dominant
share or phase coherence, fewer effective terms, and structured rather than
uniform cancellation. No single metric is declared a hard pass condition.
