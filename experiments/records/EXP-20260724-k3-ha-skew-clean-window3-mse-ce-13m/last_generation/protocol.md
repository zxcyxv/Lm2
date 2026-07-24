# Step-2250 last-checkpoint generation comparison

## Status

- State: completed
- Authorization: user-requested
- Evaluated checkpoint: `last.pt` at step 2250; not separately preserved by
  explicit user direction

## Comparison

Five fixed WikiText-103 validation prompts, 64 prompt tokens and 63 greedy
continuation tokens:

- sequential hard-token: commit and re-encode one token at a time
- block-3: commit three jointly decoded `K^1..K^3` tokens at a time

The primary judgment is direct inspection of grammar, contextual continuity,
and phrase repetition. Diversity and immediate-repetition metrics are
supporting diagnostics, not substitutes for text quality.

## Result

Evidence: [generation.md](generation.md) and [metrics.tsv](metrics.tsv).

- Sequential: distinct-2 `0.3226`, immediate repetition `0.1484`.
- Block-3: distinct-2 `0.4581`, immediate repetition `0.1581`.
- Sequential text had better local syntax in four of five samples, although it
  retained severe phrase loops and one sample collapsed to repeated subwords.
- Block-3 was more diverse by bigram count but repeatedly emitted malformed
  constructions such as duplicated articles and prepositions.

Status: `inconclusive` for general quality from five prompts; within this
sample, sequential generation is qualitatively better overall.
