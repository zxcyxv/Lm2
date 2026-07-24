# Step-1000 clean block-3 generation protocol

## Status

- State: completed
- Authorization: implied by the user-requested training and explicit interest
  in sentence quality inside the three-token window
- Checkpoint: fixed step 1000

## Question and comparison

Does the directly supervised clean three-slot causal tape produce coherent
three-token blocks under greedy generation?

- Sequential control: commit one token and re-encode after every token.
- Block-3: jointly decode `K hA`, `K^2 hA`, and `K^3 hA`, commit all three
  argmax tokens, then re-encode.

## Data and measures

- the same five fixed WikiText-103 validation prompts used by prior generation
  audits; 64 prompt tokens
- 63 greedy continuation tokens so the length is divisible by three
- seed 3361 selects prompts; generation itself is deterministic
- exact generated text, distinct-1, distinct-2, total immediate repetition,
  within-block immediate repetition, and between-block boundary repetition
- test split remains unread

## Success criterion

This small qualitative audit supports an early within-window quality signal
only if block-3 within-block repetition is below 0.20 and distinct-2 is at
least 0.50. Five prompts cannot establish general sentence quality.

## Result

Evidence: [metrics.tsv](metrics.tsv) and [generation.md](generation.md).

- Sequential: distinct-2 `0.1452`, immediate repetition `0.6194`.
- Block-3: distinct-2 `0.4613`, immediate repetition `0.2613`.
- Block-3 within-window repetition was `0.3095`; between-window boundary
  repetition was `0.1600`.

Status: `inconclusive`. Block-3 is materially less collapsed than the
step-1000 sequential control, but it misses both registered thresholds and the
text remains repetitive and syntactically broken. This is an early relative
improvement, not evidence of acceptable sentence quality.
