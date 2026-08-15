# Projection residual Gaussian audit

The audit measured `32768` self-projection residuals
`Encode(prefix, argmax Decode(KhA)) - KhA`; no gold future was used.

- Isotropic white Gaussian rejected by registered gates: **True**
- General shrinkage-Gaussian compatible by registered gates: **False**
- residual mean/RMS fraction: `0.389724`
- covariance effective-rank fraction: `0.085200`
- top/mean covariance eigenvalue: `171.829360`
- median absolute projected excess kurtosis:
  `1.366057`
- lag-1 coordinate correlation: `-0.017338`
- held-out NLL/dim, isotropic / diagonal / shrinkage-full:
  `-0.516102 / -0.764708 / -1.446664`
- full Mahalanobis mean/variance ratios:
  `0.721373 /
  163.307466`

Passing compatibility gates would not prove exact Gaussianity. Failing them
identifies which Gaussian assumption is unsuitable and whether a structured,
conditional, colored, or heavy-tailed innovation model is required.

## Interpretation

The decode--argmax--reencode projection is internally valid:
`self_token_recovery_accuracy = 1.0` over 32,768 anchors and the model chose
1,530 distinct tokens. The measured residual is therefore the requested
self-projection error, not a gold-token mismatch.

Its mean is not negligible: mean RMS is 38.97% of total residual RMS. After
centering, covariance remains strongly structured. Effective rank is only
76.34 of 896, the leading covariance eigenvalue is 171.83 times the mean,
and the top ten covariance eigenvectors explain 50.70% of centered energy.
Diagonal and shrinkage-full covariance models improve held-out NLL by 0.249
and 0.682 nat/dimension respectively, so an isotropic covariance is
decisively inadequate.

A single anisotropic Gaussian is also inadequate. Random projections have
median absolute excess kurtosis 1.366 and positive mean excess kurtosis
1.791. Under the held-out shrinkage-full fit, squared Mahalanobis norms have
only 0.721 of the expected mean but 163.31 times the chi-square variance.
This is the signature of a sharply heterogeneous or mixed population rather
than one Gaussian cloud.

Temporal coordinate correlations at lags 1, 2, 4, and 8 are all small
(`-0.017`, `0.014`, `0.027`, `0.018`). Thus the audit does not find strong
linear short-lag coloring, even though it rejects Gaussianity and isotropy.
Residual norm also has modest dependence on proposal entropy (`-0.167`) and
confidence (`0.137`), motivating token/context-conditioned grouping.

The residual is small in magnitude—mean relative RMS 2.86%, p95 4.13%—and
is not monopolized by one native spectral pair: the largest pair has 1.20%
of residual energy and the largest ten have 8.91%. The low covariance rank
therefore lies in correlated combinations of pairs rather than one isolated
frequency coordinate.

For this checkpoint, a solver based on additive correction is empirically
motivated, but a single white Gaussian corruption kernel is not. The next
model should first separate the deterministic mean correction and then test
a conditional low-rank Gaussian, Gaussian mixture, or heavy-tailed residual
model. A self-generated multi-step audit is still required before making a
claim about a 512-position accumulated innovation process.
