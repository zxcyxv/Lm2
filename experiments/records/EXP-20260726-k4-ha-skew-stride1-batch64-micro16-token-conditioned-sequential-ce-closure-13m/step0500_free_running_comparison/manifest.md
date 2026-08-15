# Step-500 token-conditioned free-running comparison

## Status

- State: preregistered before generation
- Authorization: user-requested sentence-quality and accuracy comparison
- Test split remains unread
- Current checkpoint:
  `outputs/experiments/EXP-20260726-k4-ha-skew-stride1-batch64-micro16-token-conditioned-sequential-ce-closure-13m/step0500.pt`
- Current checkpoint SHA-256:
  `a3daff6bbe01e41633ac7e9f9a06557cae9eb4dbd5572a2f977f5239419675e6`
- Architecture-control checkpoint:
  `outputs/experiments/EXP-20260725-k4-ha-skew-stride1-batch64-micro16-selective-innovation-tape-three-trajectory-prior-tau005-h1-online-mse-13m/step0500.pt`
- Architecture-control SHA-256:
  `0771b0e1f42f319bd1e1d2e85fd4c5c49c6a5b6fdad4258fc394835684622128`

## Question

Does the step-500 token-conditioned model retain sentence quality and
next-token accuracy when it must feed its own selected token back into the
recurrent transition, rather than receiving the validation continuation by
teacher forcing?

The earlier step-500 training metrics are not an answer: horizons 2--4 consume
gold previous tokens. This audit fixes one real validation prompt, generates a
token, passes that exact generated token to `U`, and repeats without reading
the reference continuation.

## Policies

All policies receive the same prompt and never inspect the reference
continuation.

1. `sequential_recurrent_greedy`: encode the prompt once, select each token by
   argmax, feed that exact token to `U`, and carry the resulting recurrent
   state for all 64 generated positions.
2. `sequential_recurrent_sample_t1`: the identical recurrence, except that the
   token is sampled from the full temperature-1 softmax with a fixed CUDA
   generator. The sampled token passed to `U` is exactly the token appended to
   the sequence.
3. `sequential_reanchored_greedy`: use the same step-500 weights, but re-encode
   the entire generated prefix before every next-token decision and apply
   `K h_t`. This is the canonical same-model AR diagnostic; it does not use
   `U` to carry state.
4. `selective_prior_ar_greedy`: use the registered previous architecture's
   step-500 checkpoint, re-encode after every generated token, sample its
   registered three innovations with a fixed generator, select the
   prefix-prior argmax branch, and greedily decode one token.

The canonical same-model comparison isolates recurrent state drift. The
selective control is an architecture comparison and has a different
transition/loss, so it is not used as an exact attribution control.

## Fixed evaluation

- WikiText-103 validation split only
- seed 1337
- validation-start generator seed `1337+2024`
- token-sampling generator seed `1337+5001`
- selective-innovation generator seed `1337+5002`
- 64 fixed examples
- prompt length 64 tokens
- generated continuation length 64 tokens
- first 12 examples decoded for qualitative inspection
- strict float32; TF32 disabled
- repository BPE tokenizer with vocabulary 8192
- CUDA evaluation microbatch 8, run after the registered training process
  releases the GPU

## Metrics

Machine-readable metrics are TSV and decoded text plus interpretation are
Markdown.

- first generated-token reference accuracy
- 64-position reference token accuracy and exact matching-prefix length
- distinct-1/2/4
- immediate, period-2, and period-4 repetition
- repeated bigram and four-gram coverage
- longest identical-token run and collapsed-sample rate
- recurrent versus reanchored token agreement
- on the recurrently generated history, canonical-reencoding versus recurrent
  next-token forward KL and top-1 agreement at positions 1, 2--4, 5--16, and
  17--64
- wall time

Reference accuracy after the first mismatch is descriptive exact-continuation
agreement, not conditional language-model accuracy: a different plausible
token changes the valid future.

## Registered interpretation

The recurrent mechanism retains short-range same-model AR behavior only if:

- position-1 recurrent/canonical logits agree within numerical tolerance;
- mean canonical-to-recurrent KL over positions 2--4 is at most 0.25; and
- recurrent/canonical top-1 agreement over positions 2--4 is at least 0.60.

It retains usable 64-token greedy sentence behavior relative to the same-model
reanchored policy only if all of:

- distinct-2 retention is at least 80%;
- immediate-repeat increase is no greater than 0.05 absolute;
- collapsed-sample rate increase is no greater than 0.10 absolute;
- reference token accuracy is at least 60% of the reanchored value; and
- the decoded samples do not show a systematic fixed-token or short-period
  cycle.

Temperature-1 sampling is a separate distributional stress test. Its exact
reference accuracy is not compared directly with greedy policies. Evidence
for long-range recurrent equivalence requires canonical-to-recurrent KL not
to grow catastrophically in the 17--64 bucket; no post-hoc numerical threshold
is assigned to that previously unmeasured regime.
