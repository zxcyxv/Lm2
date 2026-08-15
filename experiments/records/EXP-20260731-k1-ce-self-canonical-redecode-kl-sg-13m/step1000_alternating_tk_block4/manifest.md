# Step-1000 raw AR versus alternating K--T block-4 audit

## Status

- State: preregistered before rollout measurement
- Parent checkpoint: `step1000.pt`
- Checkpoint SHA-256:
  `279561b31a94dea8d5d54bbaaaf3a8b139c8c288ae6e6ee492cc375317dfa27d`
- Validation-start SHA-256:
  `7dfbbbe6d3c4393b13459a4db116bbca9e429a5544e596c4035f7a140987e1e0`
- Split: validation only; test remains unread

## Question

Does the stop-gradient self-redecode KL model fail only to match its
self-selected canonical state in one step, or does the requested alternating
`K -> T -> K -> T` recurrence introduce additional error over four tokens?

## Policies

Both policies start from the same literal gold block-boundary state and read
no corpus future token within the block.

1. `raw_ar`: compute raw `q=K(h)h`, decode and greedily select one token,
   canonically re-encode that selected token, then repeat.
2. `alternating_tk`: compute `q=K(h)h`, set `h=T(q)q`, and repeat four
   times without token feedback; jointly decode the four corrected states.

The inherited projected-shell policy is retained only as a diagnostic and is
not part of the requested inference contract.

## Fixed evaluation

- frozen step-1000 weights; no optimizer update
- 128 preserved validation sequences
- context 256
- anchors 0, 4, ..., 252
- 8,192 independent blocks and 32,768 scored tokens
- greedy actions
- strict float32; TF32 disabled
- CUDA microbatch 4

## Metrics and interpretation

Report overall and horizon-specific held-out NLL, PPL-like exponentiation,
accuracy, raw-AR/T-K token agreement, exact-block agreement, and projected
shell diagnostics. These are self-fed rollout diagnostics rather than
standard teacher-forced perplexities.

The preregistered success thresholds inherited from the parent experiment
are at least `0.90` raw-AR/T-K token agreement and `0.75` exact-block
agreement. Compare h1 agreement with overall agreement to separate initial
canonicalization error from recurrent compounding.

Numeric metrics are TSV and interpretation is Markdown.
