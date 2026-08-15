# Epoch-2 sentence-quality interpretation

## Scope

Eight fixed WikiText-2 validation prompts were used. Grassmann generated
64 tokens with greedy autoregression; the latent model generated 64 tokens
with target-free prior-argmax block 4. Both checkpoints completed two epochs
under a 20-epoch cosine schedule. This is a qualitative audit, not a robust
population estimate.

## Observed failure modes

Grassmann is locally more grammatical but collapses across prompts onto the
same phrase template, most often variants of `the first time, and the first
time`. Six of eight samples meet the registered collapse criterion. Its high
repeated-bigram and repeated-4-gram coverage (`0.974` and `0.943`) confirms
that low immediate-token repetition alone understates the collapse.

The latent block-4 output avoids a single universal phrase and has higher
distinct-2/distinct-4 (`0.361`/`0.705` versus `0.125`/`0.174`). This diversity
does not amount to good prose: doubled function words, malformed subwords,
and within-block fragments are common. Immediate repetition is `0.268` and
period-4 repetition is `0.423`, both worse than Grassmann's `0.056` and
`0.221`. Its dominant failure is broken local syntax plus block-phase
repetition rather than one global sentence template.

Reference-token accuracy is `0.0449` for latent block-4 and `0.0332` for
Grassmann AR, but eight prompts and mostly frequent function words make this
difference insufficient evidence of better language modeling.

## Bottom line

Neither epoch-2 checkpoint produces acceptable sentences. Grassmann has
better local phrase shape but severe mode collapse; latent block-4 has more
lexical/phrase diversity but substantially worse grammatical continuity.
The 24-stage latent model also took `1.41s` for the registered batchwise
generation versus `0.90s` for Grassmann AR, so this depth-matched
configuration does not demonstrate the intended wall-clock block-generation
advantage.
