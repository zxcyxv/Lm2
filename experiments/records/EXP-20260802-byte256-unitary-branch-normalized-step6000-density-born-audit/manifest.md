# EXP-20260802 step-6000 density and Born-interference audit

## Status

- State: completed
- Source checkpoint: completed step-6000 branch-normalized H16 model
- Validation split only; test remains unread

## Questions

1. Does the late-horizon space/`e` collapse coincide with the memory density
   spectrum collapsing into one mode, or does memory retain multiple modes?
2. In exact complex measurement energy, how much comes from independent write
   energies versus positive and negative pairwise cross terms, and how does that
   differ for correct shallow bytes and late collapsed predictions?

## Fixed protocol

- seed 1337 + 999; same first eight validation windows and final stride-16
  anchor as prior interpretability audits
- horizons 1--16; online step-6000 checkpoint; no model mutation
- treat the eight independent head memories as a direct-sum amplitude operator
- obtain density eigenvalues from squared singular values of each head memory,
  normalized over all heads; record purity, von Neumann entropy and effective
  rank. `SS^dagger` and `S^dagger S` share these nonzero normalized eigenvalues
- decompose the normalized query measurement amplitude by originating write,
  `m=sum_n psi_n`, and verify
  `||m||^2=sum_n||psi_n||^2 + sum_(n!=m)<psi_n,psi_m>`
- split pairwise cross terms into positive constructive and negative destructive
  mass; record normalized Born weight
- join target/predicted byte correctness and report horizon, correct/incorrect
  H1--H4, correct non-space H1--H4, and late H8--H16 space-prediction groups

## Success and evidence boundary

- density eigenvalues sum to one and Born reconstruction error is at most `1e-5`
- high purity/low effective rank indicates spectral concentration, not semantic
  identity with a space concept
- positive/negative complex cross terms are genuine amplitude interference, but
  association with correctness remains observational rather than causal
