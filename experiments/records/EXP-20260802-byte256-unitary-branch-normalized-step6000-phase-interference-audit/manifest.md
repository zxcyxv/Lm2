# EXP-20260802 step-6000 channel phase/interference audit

## Status

- State: completed
- Source checkpoint: step 6000 of the microbatch-64 continuation
- Validation split only; test remains unread

## Questions

1. Did the learned memory frequencies form a non-aliased age code over H16,
   and do transported query/key coefficient phases show age- or write-selective
   coherence?
2. At the exact output-correction level, how much constructive and destructive
   interference occurs across `(write, head, key-channel)` contributions, and
   is the result selective or broadly averaged?

## Fixed protocol

- seed 1337 + 999; the same first eight validation windows and final stride-16
  anchor used by the prior interpretability audits
- checkpoint and model remain unmodified; `gamma` is absent and is not inferred
- expand every write as `k_n v_n^dagger`, transport it by
  `exp(-i age theta_c)`, and decompose every measurement correction by write,
  head, and key channel
- verify the complete channel decomposition against the model correction
- record learned `theta_c`, phase-code Gram similarities for ages 0--15,
  transported coefficient magnitude/phase/cosine/coherence, exact signed
  projection onto the total correction, and contribution norm
- aggregate positive projection mass, negative projection mass, cancellation,
  effective channel-write count, current-write share, and dominant share

## Success criteria and evidence boundary

- maximum exact correction reconstruction error at most `1e-5`
- an age-resolving frequency bank should have low off-diagonal phase-code Gram
  similarity; selective interference should have low effective contribution
  count and non-uniform dominant shares
- coefficient `cos(phase)` is the scalar query/key phase diagnostic, but values
  and the learned output map are complex. Therefore constructive/destructive
  claims about the actual hidden update use signed hidden-space projections,
  not coefficient cosine alone.
