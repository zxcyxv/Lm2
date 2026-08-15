# Step-1000 prior-trajectory block-4 generation

## Status

- State: preregistered before generation
- Authorization: user-requested sentence-quality comparison
- Checkpoint step: 1000
- Checkpoint SHA-256:
  `f56abdf9413e3df11c2ff0b3cf4ba2c75bcb627116b6fbf493331d9fc7d4ea07`
- Test split remains unread

## Question

At the completed initial-condition trajectory checkpoint, how much
free-running text quality is lost when four tokens are decoded in parallel
and committed before re-encoding, rather than re-grounding after every token?
Does the learned prefix prior and persistent trajectory improve generation
relative to the clean K orbit?

## Matched generation policies

All policies use greedy token decoding and the same prompts.

1. `clean_ar`: use `K hA`, commit one token, then re-encode.
2. `clean_block4`: use `K hA ... K^4 hA`, commit four tokens, then
   re-encode.
3. `prior_ar`: compute the prefix prior, select its argmax trajectory, use
   `K hA + eta_i`, commit one token, then recompute the prior.
4. `prior_block4`: compute the prefix prior once, select its argmax
   trajectory, propagate that initial condition with K for four horizons,
   commit all four tokens, then recompute the prior.

No gold continuation, posterior responsibility, confidence selector, beam
search, sampling, rejection, or post-generation correction participates in
selection. Thus the AR/block-4 comparison within each clean/prior pair differs
only in the re-grounding interval.

## Fixed evaluation

- WikiText-103 validation split only
- 64 fixed examples
- prompt length 64 tokens
- generated continuation length 64 tokens
- starts sampled by `torch.Generator(seed=1337+2024)`
- qualitative report contains the first eight fixed examples
- strict float32, TF32 disabled
- tokenizer: repository BPE vocabulary 8192

## Metrics

Metrics are written to TSV; decoded text and qualitative interpretation are
kept in Markdown.

- reference-position token accuracy, including within-block offsets 1--4
- distinct-1/2/4
- immediate, period-2, and period-4 token repetition
- exact adjacent four-token block repetition
- within-block versus block-boundary immediate repetition
- longest identical-token run and collapsed-sample rate
- wall time and number of encoder/re-grounding calls
- for prior policies, branch usage, prior maximum, entropy, and branch-switch
  rate

## Registered interpretation thresholds

The primary comparison is `prior_block4` against `prior_ar`. Evidence that
block-4 retains usable sentence quality requires all of:

- distinct-2 retention of at least 80%;
- immediate-repeat increase no greater than 0.05 absolute;
- period-4 repeat increase no greater than 0.10 absolute;
- collapsed-sample rate increase no greater than 0.10 absolute;
- reference token accuracy at least 60% of the matched AR value; and
- no systematic four-token cycle in the eight decoded qualitative samples.

These thresholds diagnose degeneration, not human-equivalent prose quality.
A positive architecture claim additionally requires the decoded samples to
remain locally grammatical and topically connected to their prompts.
The clean pair is an internal control and the comparison remains limited to
this checkpoint, validation sample, and greedy policy.
