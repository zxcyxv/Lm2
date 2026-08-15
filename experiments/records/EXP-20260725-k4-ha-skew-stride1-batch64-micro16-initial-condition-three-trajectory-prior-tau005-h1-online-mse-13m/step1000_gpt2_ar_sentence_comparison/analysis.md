# Analysis: matched GPT-2 versus latent-orbit generation

## Outcome

The user's calibration is supported: the parameter-matched GPT-2 does not
produce consistently good prose after 1000 updates. It often begins with a
locally plausible continuation and then falls into a frequent WikiText
heading or phrase template. Poor absolute prose at this training horizon is
therefore not, by itself, evidence against parallel latent rollout.

The baseline is still informative because the failure signatures differ.
Standard GPT-2 primarily exhibits **phrase/template attraction**. The clean
latent policies exhibit much stronger **single-token or low-cardinality
attraction**. The learned initial condition suppresses that fixed-point
collapse, but block-4 commitment exposes a separate **four-position limit
cycle**. Thus the initializer improved diversity without yet producing a
coherent four-token sentence trajectory.

The numerical evidence is in [`metrics.tsv`](metrics.tsv), paired outcomes
are in [`paired_vs_gpt2.tsv`](paired_vs_gpt2.tsv), and the first 16 fixed
examples are in [`generation.md`](generation.md).

## Aggregate pattern

| policy | distinct-2 | immediate | period-2 | period-4 | exact aligned block repeat | collapsed | mean longest run |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 AR | 0.2768 | 0.0930 | 0.1923 | 0.2557 | 0.1167 | 0.2969 | 3.70 |
| clean AR | 0.1791 | 0.5801 | 0.5602 | 0.6435 | 0.5333 | 0.7031 | 33.92 |
| prior AR | 0.2634 | 0.3867 | 0.3727 | 0.3852 | 0.3333 | 0.4688 | 22.16 |
| clean block-4 | 0.2773 | 0.5511 | 0.4461 | 0.6617 | 0.4583 | 0.5781 | 22.16 |
| prior block-4 | 0.3380 | 0.2299 | 0.0640 | 0.4951 | 0.2687 | 0.1719 | 4.19 |

`collapsed` means either an identical-token run of at least eight or
distinct-1 below 0.1. It therefore includes low-vocabulary phrase loops as
well as literal one-token collapse.

## What the GPT-2 baseline establishes

GPT-2 is visibly undertrained:

- 19 of 64 continuations meet the registered collapse heuristic.
- Repeated four-gram coverage is 0.7395 even though immediate repetition is
  only 0.0930.
- Typical continuations move into high-frequency templates such as
  `= = =`, `the first time`, or repeated `@-@ year` structures.
- In the qualitative examples, GPT-2 often preserves the prompt's topic and
  ordinary subject/verb/preposition order for an initial clause, but it does
  not sustain a factual or grammatical continuation for 64 tokens.

Its median immediate-repeat rate is zero and its median longest identical run
is one. Its main failure is consequently not the same-token fixed point seen
in the clean latent model; it is recycling a short syntactic or corpus-format
template. This is the useful quality floor supplied by the baseline.

## What is specific to the latent model

### Clean orbit: a strong decoder-token attractor

`clean_ar` collapses on 45 of 64 examples. Its median longest identical run
is 39 tokens, versus one for GPT-2. The qualitative failures (`States`,
`singer`, `video`, `Army`, `NY`, punctuation) show that repeated
re-encoding does not rescue the clean `K hA` proposal once the generated
prefix enters a narrow decoder/LM-head basin.

`clean_block4` has almost the same distinct-2 as GPT-2, but that aggregate is
misleading: immediate repetition is 0.5511 and exact aligned block
repetition is 0.4583. Four different output slots can raise n-gram diversity
while the continuation remains structurally broken.

### Initial-condition prior: fixed point becomes a four-phase cycle

The strongest result is the lag signature:

- GPT-2 AR: lag-1 `0.0930`, lag-2 `0.1923`, lag-4 `0.2557`.
- prior block-4: lag-1 `0.2299`, lag-2 `0.0640`, lag-4 `0.4951`.

For GPT-2, repetition rises gradually with lag because short phrases recur.
For `prior_block4`, lag 2 is unusually low while lag 4 is high. Positions in
the same block phase repeat, while neighboring phases remain different.
Aligned adjacent four-token blocks repeat at 0.2687, versus 0.1167 for
GPT-2.

The block-boundary diagnostic says the same thing. GPT-2 has nearly equal
immediate repetition within blocks and across the arbitrary four-token
boundary (`0.0915` versus `0.0979`). `prior_block4` has `0.2757` within a
predicted block but only `0.0833` across re-encoding boundaries. The
four-token boundary is therefore visible in the output statistics even
though it has no linguistic meaning.

This explains the superficially favorable diversity numbers:

- `prior_block4` distinct-2 is 0.3380, higher than GPT-2's 0.2768.
- its collapse heuristic fires on 11 examples, fewer than GPT-2's 19;
- its prompt-token reuse is 0.6538, versus 0.4846 for GPT-2.

But the decoded text repeatedly uses slot-like forms such as `the the X of`,
punctuation in a fixed phase, and four-position noun/function-word
templates. Lexical reuse is often just repetition of a prompt noun
(`video`, `team`, `storm`, `British`) rather than a coherent proposition.
Reference accuracy is also only 0.0254 versus 0.0186, much too small and too
free-running-dependent to establish superior language modeling.

## Architectural reading

The initializer appears to have accomplished its narrow geometric job.
Persistent branch separation prevents all four horizons from falling into
one identical token, and committing the branch for the whole block produces
more token variety than reselecting it every step.

What it has not established is **semantic trajectory commitment**. The
training cost for a branch is a sum of four tokenwise cross-entropies. Sharing
the branch identity makes the assignment block-level, but each horizon can
still be solved as a phase-specific decoder classification problem. A stable
four-slot code is therefore a legal shortcut:

`branch commitment + K phase identity != coherent future commitment`.

The comparison suggests the current failure hierarchy:

1. the clean orbit tends toward a token-level fixed point;
2. the initial-condition residual separates the horizon states and weakens
   that fixed point;
3. the all-horizon classifier uses the separation to form four stable output
   phases;
4. no present objective forces those four phase outputs to form one
   linguistically coherent block.

This is not evidence that a simplex map or an iterative canonicalizer is
needed. It is evidence that geometric branch persistence and marginal
all-horizon correctness are insufficient proxies for joint trajectory
semantics.

## Fairness limits

The comparison is deliberately descriptive.

- The models use byte-identical train/validation data, the same tokenizer,
  seed, context length, effective batch, 1000 updates, and nearly identical
  parameter counts.
- GPT-2 is trained with one-step CE at every context position, exactly
  matching AR inference. It is not literally supervised on only one token
  per window; it receives 256 next-token labels per example.
- The latent model scores four overlapping horizons and three branches,
  adds h1 state and prior losses, and is optimized on a 6000-update schedule
  observed at step 1000. GPT-2 uses a schedule designed to finish at update
  1000.
- Update count is consequently neither a token-loss-event match nor a
  compute match. The comparison cannot assign the entire quality gap to
  architecture.
- Greedy generation amplifies modal collapse in both models. Sampling or
  beam search would answer a different question.

Within those limits, the evidence supports two simultaneous conclusions:
expecting polished prose from either 13M model at 1000 updates is
unreasonable, and the latent model still has an architecture-specific
four-phase coherence problem that the undertrained GPT-2 does not exhibit in
the same form.
