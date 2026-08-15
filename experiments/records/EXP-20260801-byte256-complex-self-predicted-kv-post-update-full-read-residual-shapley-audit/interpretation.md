# Interpretation

This audit uses the model's own greedy-AR canonical hidden as the reference. Held-out continuation states are not the agreement target.

## Innovation versus canonical AR residual

| H | Agreement | Prior rel. MSE | Full rel. MSE | Error reduction | Helpful rows | Residual cosine | Global scale | Global oracle MSE | Rowwise oracle MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000000 | 0.692875 | 0.274246 | 0.418629 | 0.997314 | 0.773594 | 1.126824 | 0.268875 | 0.258025 |
| 2 | 0.466553 | 0.779062 | 0.785615 | -0.006552 | 0.511963 | 0.126890 | 0.427831 | 0.770753 | 0.742262 |
| 3 | 0.152832 | 0.935681 | 0.994058 | -0.058377 | 0.248779 | -0.029697 | -0.265079 | 0.933000 | 0.900358 |
| 4 | 0.107422 | 1.016923 | 1.118897 | -0.101974 | 0.118896 | -0.134177 | -0.860211 | 0.989186 | 0.955447 |
| 2-4 | 0.242269 | 0.910556 | 0.966190 | -0.055634 | 0.293213 | -0.012328 | -0.189509 | 0.909107 | 0.866022 |

## Agreement stratification over H2--H4

| Group | Rows | Full rel. MSE | Innovation reduction | Residual cosine | AR margin | Max logit delta |
|---|---:|---:|---:|---:|---:|---:|
| match | 2977 | 0.783386 | -0.008763 | 0.118154 | 0.691589 | 5.522811 |
| mismatch | 9311 | 1.024638 | -0.070620 | -0.054047 | 0.459972 | 9.241262 |

## Head-level exact Shapley allocation over H2--H4

| Head | Mean error reduction | Positive rows | Residual cosine | Norm ratio |
|---:|---:|---:|---:|---:|
| 5 | -0.003244 | 0.452230 | 0.035909 | 0.053836 |
| 4 | -0.005138 | 0.409587 | 0.008632 | 0.054142 |
| 3 | -0.006677 | 0.302816 | -0.015836 | 0.049405 |
| 7 | -0.007106 | 0.420166 | 0.004804 | 0.056912 |
| 0 | -0.007246 | 0.337402 | -0.023348 | 0.047906 |
| 2 | -0.007394 | 0.414551 | 0.007544 | 0.056778 |
| 6 | -0.008726 | 0.394287 | -0.000678 | 0.054079 |
| 1 | -0.010103 | 0.240316 | -0.029829 | 0.056510 |

## Same-weight recurrent-form counterfactual

| Mode | H2 | H3 | H4 | H2--H4 | Exact block |
|---|---:|---:|---:|---:|---:|
| post-update-full-read | 0.466553 | 0.152832 | 0.107422 | 0.242269 | 0.009277 |
| split-query-residual | 0.312012 | 0.147949 | 0.120850 | 0.193604 | 0.009521 |

## Structural checks

- additive reconstruction max error: 3.815e-06
- head Shapley sum max error: 3.576e-07
- non-tied flip classification errors: 0
- counterfactual AR-reference max logit error: 0.000e+00

## Conclusion

The coherent post-update full reread is not the cause of the regression. At
the same trained weights it raises H2--H4 agreement from `0.193604` to
`0.242269`, with identical H1 central logits and an identical greedy-AR
reference.

The failure is horizon-dependent innovation geometry:

- H1 is a successful correction: mean relative MSE falls by `0.418629`, and
  `99.73%` of rows improve.
- H2 has a weakly useful direction but excessive fixed magnitude. Its one
  global least-squares scale is `0.427831`, yet coefficient one slightly
  increases mean error.
- H3 and H4 have global optimum scales `-0.265079` and `-0.860211`. The write
  has become directionally wrong, not merely too large.
- One fixed positive temperature cannot repair a coefficient whose preferred
  sign changes with horizon.

Mismatch rows also have substantially worse recurrent geometry than matching
rows: full relative MSE is `1.024638` versus `0.783386`, read-state cosine is
`0.348962` versus `0.674864`, and median logit displacement is `9.241263`
versus `5.522811`. Their AR margins are narrower (`0.459972` versus
`0.691589`), so decision-boundary sensitivity contributes, but the dominant
evidence is a recurrent-state divergence rather than a pure decoder-margin
effect.

Thus H1 regression learned a locally useful innovation on the one-step state
distribution, but that innovation is not closed under its own repeated
rollout. The next mechanism change should target multi-step self-composition
of the innovation direction, not polar/L2 constraints or a single scalar
coefficient.
