# Window-1 final matched sequential-generation protocol

## Status

- State: planned
- Authorization: user-requested
- Checkpoint: NLL-selected/final step 6000

## Question and comparison

What is the actual greedy sequential-generation quality of the previous
window-1 `hA` skew-K model? The evaluation uses the exact prompts and
continuation length of the step-1000 window-3 generation audit so the texts and
simple repetition metrics are directly comparable.

## Data and measures

- five fixed WikiText-103 validation prompts selected with seed 3361
- 64 prompt tokens and 63 greedy continuation tokens
- one-token generation with real re-encoding after every committed token
- distinct-1, distinct-2, immediate repetition, and exact generated text
- no inference-time noise; test split remains unread

## Interpretation criterion

Immediate repetition below 0.20 and distinct-2 above 0.30 count as evidence
against token-level collapse on these prompts. Repeated phrases and factual or
syntactic quality are inspected separately; five prompts cannot establish
general language quality.
