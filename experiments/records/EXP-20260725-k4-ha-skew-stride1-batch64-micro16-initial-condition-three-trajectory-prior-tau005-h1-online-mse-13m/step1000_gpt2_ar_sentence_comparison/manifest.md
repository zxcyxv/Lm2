# Step-1000 matched GPT-2 AR sentence-pattern comparison

## Status

- State: registered before the 64-sample cross-model evaluation
- Authorization: user-requested sentence-quality comparison
- Evidence class: post-hoc descriptive audit
- Chronology caveat: the four latent-orbit policies and five legacy GPT-2
  samples had already been inspected; the remaining 59 GPT-2 continuations
  and aggregate cross-model metrics had not
- Test split remains unread

## Completion

- Evaluation completed successfully for all 64 registered prompts.
- The emitted `starts.tsv` has the preregistered SHA-256 and is byte-identical
  to the earlier latent-only evaluation start list.
- Producer:
  [`../../../../eval_step1000_gpt2_ar_sentence_comparison.py`](../../../../eval_step1000_gpt2_ar_sentence_comparison.py)
- Metrics: [`metrics.tsv`](metrics.tsv) and
  [`sample_metrics.tsv`](sample_metrics.tsv)
- Decoded evidence: [`generation.md`](generation.md)
- Interpretation: [`analysis.md`](analysis.md)

## Question

On the same 64 WikiText-103 validation prompts, which degeneration patterns
are shared by a parameter-matched, 1000-update standard GPT-2 AR model and
which are specific to the latent-orbit model or its four-token commitment?

This comparison does not assume that equal parameter count and update count
imply equal supervision. GPT-2 receives ordinary dense one-step
teacher-forced CE, while the latent-orbit model receives overlapping
four-horizon trajectory supervision plus auxiliary state and prior losses.
The baseline is therefore a sentence-pattern and undertraining sanity
control, not an architecture-isolated causal ablation.

## Checkpoints

### Latent orbit

- experiment:
  `EXP-20260725-k4-ha-skew-stride1-batch64-micro16-initial-condition-three-trajectory-prior-tau005-h1-online-mse-13m`
- step: 1000
- parameters: 13,087,332
- SHA-256:
  `f56abdf9413e3df11c2ff0b3cf4ba2c75bcb627116b6fbf493331d9fc7d4ea07`

### Standard GPT-2 AR

- source experiment:
  `/workspace/Lm/experiments/records/EXP-20260723-gpt2-baseline-matched-13m`
- architecture: standard pre-LayerNorm GPT-2, learned absolute positions,
  tied token/LM-head embedding, width 440, 4 layers, 8 heads
- step: 1000 (`best.pt`; best step and last step coincide)
- parameters: 13,033,680
- SHA-256:
  `bfff4c3f369b034fd979bcec564a1a136c869fadc6c36b89f34c8a7d68963522`

## Data identity

Both repositories contain byte-identical tokenized data and tokenizer:

- train:
  `9d931b6823da9bd82605164f10c2c0ce518f4a75c95574fe8b4035efd2ab7ef6`
- validation:
  `eb2539b969f967b33a175842d6a7398de0b52265ef012d16f5dbc6f221c3b9a2`
- tokenizer:
  `1155e79411f38a19d40bf13e32e5cdf38cbfc6d8fdb999df057650b70aacf590`

## Compared generation policies

All policies use greedy decoding from the same prompts.

1. `gpt2_ar`: standard GPT-2, re-evaluated after each committed token.
2. `clean_ar`: latent model's clean `K hA`, one token per re-encoding.
3. `prior_ar`: prior-selected persistent branch, one token per re-encoding.
4. `clean_block4`: clean `K hA ... K^4 hA`, four tokens per re-encoding.
5. `prior_block4`: prior-selected initial condition propagated for four
   horizons, four tokens per re-encoding.

No gold continuation, posterior selector, sampling, rejection, or
post-generation correction participates in generation.

## Fixed evaluation

- WikiText-103 validation split
- 64 examples, prompt length 64, continuation length 64
- starts from `torch.Generator(seed=1337+2024)`
- exact start list must equal the existing block-4 generation record
  (SHA-256:
  `1b17eac232f2ec8b2a8f0fd39e726cdadd56cf3b162c3ac752540c36c0f787dd`)
- strict FP32 with TF32 disabled
- BPE vocabulary 8192

## Metrics and interpretation contract

Raw aggregate and per-sample metrics are TSV. Decoded examples and
interpretation are Markdown.

- `distinct-1/2/4`, immediate repetition, period-2/4 repetition, exact
  adjacent four-token-block repetition, longest identical-token run, and
  collapsed-sample rate
- reference-position accuracy only as a weak teacher-forced-alignment
  diagnostic, not as sentence quality
- `phase_excess = period_4_repeat - immediate_repeat`: positive excess is a
  compact signal of four-position phase locking rather than simple
  same-token repetition
- within-block versus boundary immediate repetition
- prompt-vocabulary reuse and reference-vocabulary overlap as weak lexical
  anchoring diagnostics
- paired per-sample wins against `gpt2_ar`
- decoded first 16 fixed examples for qualitative inspection

The primary descriptive distinctions are:

1. generic undertraining collapse: high immediate repetition, long identical
   runs, and low distinct-2, expected to be possible in every 1000-step model;
2. phrase/template looping: repeated n-grams without necessarily repeating
   one token;
3. block-phase locking: period-4 or exact-block repetition materially above
   same-token repetition, predicted to be more specific to block-4 rollout;
4. lexical anchoring without syntax: prompt-related nouns or function-word
   templates survive while grammatical relations fail.

No universal claim about all 13M/1000-step GPT-2 models will be made from one
seed and one 64-prompt greedy sample.
