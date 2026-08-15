# Corrected process-noise effect

The correction passed its scale-zero gate exactly: the deterministic arm had
zero logit RMS, zero top-1 disagreement, and identical deterministic and
stochastic NLL when both calls used the same full-window causal encode.

At effective noise scale 0.05, the parent checkpoint's process-noise-only
effect was small: logit RMS `0.051129`, top-1 disagreement `0.001892`, and NLL
increase `0.000056`.  After 300 steps, stochastic CE reduced logit RMS to
`0.049792`; adding the ray auxiliary reduced it further to `0.049362`.
Therefore neither objective made the noise tape a stronger selector of output
modes in this continuation.

The corrected values supersede only the interpretation of the parent
`noise_logit_rms`, `noise_top1_disagreement`, and deterministic-versus-
stochastic NLL deltas.  The original training loss, gradient norm, and
within-full-path ray statistics remain valid and are not replaced.
