# Step-1000 self-projection residual Gaussian audit

## Status

- State: preregistered before reading projection residuals
- Parent checkpoint: `step1000.pt`
- Expected checkpoint SHA-256:
  `0fb701f5ccd23214ad1c500ffbf133e06aa904c718f90e0474e3058549de6481`
- Split: validation only; the test split remains unread
- Validation starts: the parent's preserved 128 starts
- Seed: 1337 for deterministic random projections and Gaussian references

## Question

When the model's clean continuous proposal is decoded greedily and that
self-selected token is re-encoded behind the identical literal prefix, is
the resulting projection residual adequately described as Gaussian noise?

No gold future token or gold future hidden state enters the residual:

`hA = Encode(prefix ending at A)`

`u = K hA`

`B_self = argmax Decode(prefix, u)`

`hB_self = Encode(prefix, B_self)`

`epsilon = hB_self - u`.

The audit concerns this checkpoint's own decode--commit--reencode error. It
does not claim to measure an intrinsic residual distribution of language.

## Sampling

- 128 preserved validation windows
- 256 causal anchors per window
- 32,768 residual vectors of width 896
- examples 0--63 form the fit partition
- examples 64--127 form the held-out diagnostic partition
- overlapping anchors are retained but uncertainty is summarized by
  per-example bootstrap-free aggregates rather than treating every anchor as
  an independent document

## Registered diagnostics

- exact dense action re-encoding versus literal prefix re-encoding gate
- self-token recovery after re-encoding
- residual mean, RMS, relative RMS, and cosine geometry
- covariance eigenvalue spectrum and effective rank
- held-out isotropic, diagonal, and shrinkage-full Gaussian NLL
- random-projection skewness and excess kurtosis
- whitened Mahalanobis squared norm mean and variance relative to
  `chi-square(width)`
- lag-1, lag-2, lag-4, lag-8 residual correlation within windows
- residual-norm dependence on proposal entropy and confidence
- spectral-pair residual energy concentration

## Interpretation gates

`epsilon ~ N(0, sigma^2 I)` is rejected as a useful model if any of:

- the residual mean RMS exceeds 10% of total residual RMS
- covariance effective rank is below 50% of width
- top covariance eigenvalue exceeds 5 times the mean eigenvalue
- median absolute random-projection excess kurtosis exceeds 0.2
- absolute lag-1 correlation exceeds 0.05
- diagonal Gaussian held-out NLL improves over isotropic Gaussian by more
  than 0.05 nat per dimension

A general fitted Gaussian remains only *compatible*, not proven, if:

- shrinkage-full Gaussian improves held-out NLL over diagonal Gaussian
- whitened Mahalanobis mean and variance ratios are each within 10% of their
  chi-square expectations
- median absolute projection skewness and excess kurtosis are below 0.1

Failure of the isotropic-white model does not imply failure of conditional,
colored, low-rank, or heavy-tailed innovation models.

Metrics are written to TSV and interpretation to Markdown. Residual tensors
and checkpoint copies are not added to Git.
