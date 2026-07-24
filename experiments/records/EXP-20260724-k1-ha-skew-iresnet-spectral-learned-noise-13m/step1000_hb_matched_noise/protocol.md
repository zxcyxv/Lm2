# Step-1000 matched hB-noise decoder audit

## Status

- State: completed
- Authorization: user-requested
- Checkpoint: fixed step 1000

## Question

Is the inverse decoder's large error specific to the direction
`K hA - hB`, or does a random perturbation of `hB` with the same relative
MSE produce comparable amplification?

## Comparison, data, and seeds

- WikiText-103 validation split only; test remains unread
- the same 256 fixed prefixes used by the inline h1..h5 evaluator
- checkpoint step 1000
- actual state: `K hA`
- norm-matched control: `hB + n`, with `||n|| = ||K hA - hB||` per example
- geometry-matched control: the error component parallel to `hB` is retained
  while its orthogonal component is randomly rotated; this preserves input
  distance, perturbed-state norm, and cosine to `hB`
- eight random draws, generator seed `1337 + 8811`
- strict float32 with TF32 disabled

## Measures and decision rule

The audit records input relative MSE/cosine, decoded relative MSE/cosine,
local L2 gain, target accuracy, NLL, rank, and margin. Gold `hB` is the exact
decoder control.

- Comparable generic amplification: geometry-matched mean decoded relative
  MSE is within a factor of 0.8 to 1.25 of the actual `K hA` value, and both
  local gains exceed one.
- Direction-specific amplification: the ratio falls outside that interval.
- Claims remain limited to this checkpoint, split, and perturbation scale.

## Result

Evidence: [analysis.md](analysis.md), [summary.tsv](summary.tsv), and
[diagnostics.tsv](diagnostics.tsv).

- All three perturbed variants had mean input relative MSE `0.0217294`.
- Actual `K hA`: decoded relative MSE `18.0164`, local L2 gain `1.18048`,
  accuracy `0.226562`.
- Geometry-matched random noise: decoded relative MSE `43.2314`, local L2 gain
  `1.81281`, accuracy `0.446289`.
- The geometry-matched/actual decoded relative-MSE ratio was `2.39955`.
- Gold `hB` decoded with accuracy `1.0`; embedding roundtrip maximum error was
  `8.75e-7`.

Status: `supported` within this audit that amplification is not specific to the
`K hA-hB` direction. Equal-geometry random perturbations were amplified more,
so the registered direction-specific alternative is refuted. The much worse
token accuracy of `K hA` despite its smaller decoded L2 error separates the
semantic/head direction problem from perturbation magnitude.
