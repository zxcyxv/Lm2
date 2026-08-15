# EXP-20260802 step-6000 scaling/Jacobian audit

## Status

- State: completed
- Source checkpoint: step 6000 of the microbatch-64 continuation
- Validation split only; test remains unread

## Question and comparison

At step 6000, do the four assumptions previously audited at steps 100--1000
become more or less accurate?

1. centered latent residual grows approximately as `sqrt(r)`;
2. residual direction change decays approximately as `1/sqrt(r)`;
3. the complete H16 directional Jacobian retains a near-lossless highway;
4. coupling `E=J-D` decays approximately as `1/sqrt(r)`.

## Fixed protocol

- same four fixed validation windows from seed 1337 + 999
- all sixteen stride-16 roots for trajectory summaries
- horizons 2--16 log-log fits
- four deterministic unit JVP probes on the first root
- compare the resulting TSV fields directly with the preserved step-1000 audit

## Success criterion and evidence boundary

Predicted exponents are `+0.5`, `-0.5`, near-unit H16 product gain, and `-0.5`
respectively. Directional probes are not spectral-norm guarantees, and memory
norm/token-CE correlation remains supplementary and non-causal.
