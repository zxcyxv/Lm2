# Step-1750 sequential-generation check

## Status

- State: completed
- Authorization: user-requested
- Requested checkpoint: step 1500
- Executed checkpoint: step 1750, because `last.pt` had already advanced and
  the training run did not preserve step 1500 separately
- Preserved artifact: `step1750.pt`, SHA-256
  `d74d061e355e01c35c9e4ebf55ed31b60c43c61f15d13cee9f41d44de591c679`

## Evaluation

- the same five fixed WikiText-103 validation prompts as the step-1000 audit
- 64 prompt tokens and 63 greedy continuation tokens
- primary result: one-token sequential generation with real re-encoding
- a matched block-3 result is emitted by the shared evaluator as a control
- distinct-1, distinct-2, immediate repetition, exact text, and block-position
  repetition split
- no inference-time noise; test split remains unread

## Interpretation criterion

Relative to the step-1000 sequential result, a reduction of immediate
repetition by at least 0.10 counts as a material early improvement. Five
prompts cannot establish general language quality.

## Result

Evidence: [metrics.tsv](metrics.tsv) and [generation.md](generation.md).

- Sequential distinct-1 `0.2698`, distinct-2 `0.4161`, immediate repetition
  `0.2613`.
- The step-1000 sequential control was `0.1175/0.1452/0.6194`,
  respectively.
- The generated text now forms local sentence templates, but repeatedly loops
  phrases such as “the building” and “the British Army”.

Status: `supported` for a material reduction in early token-level collapse
relative to step 1000; `inconclusive` for general sentence quality.
