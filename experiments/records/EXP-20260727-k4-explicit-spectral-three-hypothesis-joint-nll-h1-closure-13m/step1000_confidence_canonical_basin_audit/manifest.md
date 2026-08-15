# Step-1000 confidence-to-canonical-basin audit

## Status

- State: preregistered before confidence-ascent measurements
- Parent checkpoint: `step1000.pt`
- Expected SHA-256:
  `0fb701f5ccd23214ad1c500ffbf133e06aa904c718f90e0474e3058549de6481`
- Split: validation only; test remains unread
- Validation starts: all 128 preserved parent starts
- Anchors: positions 63, 127, 191, and 255
- Seed: 1337

## Question

For the model's self-selected token `B`, does increasing `p(B)` in the
continuous operator-state input force that state toward the canonical
re-encoding `hB = Encode(prefix, B)`?

For every registered anchor:

`hA = Encode(prefix ending at A)`

`u0 = K hA`

`B = argmax Decode(prefix, u0)`

`hB = Encode(prefix, B)`

The primary local diagnostic is:

`cos(grad_u log p(B|u0), hB-u0)`.

## Confidence ascent

The first 32 preserved examples (128 anchors) are optimized from three
registered starts:

1. `proposal`: `u0`
2. `canonical_perturbation`: `hB` plus a seeded random tangent perturbation
   with RMS equal to `RMS(hB-u0)`
3. `random_sphere`: a seeded random vector rescaled to `||u0||`

`B` remains fixed. Each start receives 40 Adam ascent steps on `log p(B)`
with learning rate `0.02`. After each step the state is rescaled to its
initial norm, preventing confidence growth by radial escape. Metrics are
registered at steps 0, 1, 2, 4, 8, 16, 24, and 40.

## Metrics

- canonical `p(B|hB)` and logit margin
- proposal `p(B|u0)` and distance to `hB`
- initial likelihood-gradient/canonical-residual cosine
- fraction of positive and negative initial alignments
- confidence and relative distance to `hB` by start and ascent step
- top-1 preservation for the fixed B
- minimum and median canonical distance among states reaching confidence
  thresholds `0.5`, `0.9`, `0.99`, `0.999`, `0.9999`
- existence of a high-confidence counterexample farther from `hB` than the
  original proposal

## Interpretation

The strong basin hypothesis is falsified if either:

- median initial gradient/residual cosine is non-positive; or
- any optimized state reaches `p(B) >= 0.99` while remaining farther from
  `hB` than its corresponding proposal start.

It is supported, not proven, only if:

- median initial cosine is at least `0.5`
- at least 90% of initial cosines are positive
- proposal-start ascent monotonically decreases median distance as
  confidence increases
- every observed `p(B) >= 0.99` state is closer to `hB` than `u0`.

Failure to reach 99% is reported rather than treated as support. Numeric
results are TSV and interpretation is Markdown.
