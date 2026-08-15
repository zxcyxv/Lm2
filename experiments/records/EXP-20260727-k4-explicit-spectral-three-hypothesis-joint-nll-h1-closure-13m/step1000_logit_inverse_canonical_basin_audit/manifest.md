# Step-1000 full-logit and inverse-hidden canonical-basin audit

## Status

- State: preregistered before reading the new energy trajectories
- Parent checkpoint: `step1000.pt`
- Expected SHA-256:
  `0fb701f5ccd23214ad1c500ffbf133e06aa904c718f90e0474e3058549de6481`
- Split: validation only; test remains unread
- Validation starts: all 128 preserved starts
- Anchors: 63, 127, 191, 255
- Optimization subset: preserved examples 0--31
- Seed: 1337

## Question

Does the exact-inverse geometry make canonical `hB` identifiable from the
full decoder output even though maximizing the single self-token likelihood
did not?

For `u0 = K hA`, `B = argmax Decode(u0)`, and
`hB = Encode(prefix, B)`, compare:

1. centered full-logit energy

   `E_logit(u) = mean((C logits(u) - C logits(hB))^2)`

2. inverse-decoder hidden energy

   `E_hidden(u) = mean((D(u) - D(hB))^2)`.

`C` removes each vector's vocabulary-logit mean. The previous registered
single-token confidence audit is the control.

## Solver

Each energy is optimized for 40 Adam steps at learning rate 0.02 from:

- clean proposal `u0`
- canonical `hB` plus a seeded tangent perturbation matched to
  `RMS(hB-u0)`
- seeded random state on the `||u0||` sphere

After every update each state is returned to its initial norm. Metrics are
recorded at steps 0, 1, 2, 4, 8, 16, 24, and 40.

## Registered diagnostics

- exact inverse reconstruction error `D(hB)` versus token embedding `eB`
- initial cosine between each negative energy gradient and `hB-u0`
- energy, relative canonical distance, B confidence, and B top-1 by step
- fraction of proposal trajectories with monotonically decreasing median
  canonical distance
- residual removal fraction at step 40
- distant low-energy counterexamples from random-sphere initialization

## Interpretation

An energy provides a useful local canonical lift if, from proposal starts:

- median initial gradient/residual cosine is at least 0.5
- at least 90% of initial cosines are positive
- median canonical distance decreases monotonically at registered steps
- step-40 median distance is at most 25% of step-0 distance
- fixed B remains top-1 in at least 99% of trajectories

The strong global-basin claim is not established by proposal success. It is
falsified for a tested energy if random-sphere starts reach at most 1% of
proposal-start terminal energy while retaining canonical relative distance
above 0.5.

Numeric results are TSV and interpretation is Markdown.
