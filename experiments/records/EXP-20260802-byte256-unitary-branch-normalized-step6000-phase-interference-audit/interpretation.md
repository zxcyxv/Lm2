# Interpretation

## 1. Learned frequency and phase code

The 128 learned memory frequencies span `-0.1690` to `+0.1235` radians per
step, with standard deviation `0.07265`. They began as a linear bank over
`[-0.05,+0.05]`. Mean absolute displacement was `0.04154`, while correlation
with the initial ordering remained `0.867`: training broadened the bank without
completely rearranging it.

No individual channel completes a period within 32 reasoning steps; the
shortest learned period is about 37.2 steps. The bank therefore has no literal
single-channel wraparound alias inside H16. Its distributed age-code Gram is:

| Age separation | Initial abs similarity | Learned abs similarity |
|---:|---:|---:|
| 1 | 0.9996 | 0.9974 |
| 2 | 0.9983 | 0.9896 |
| 4 | 0.9932 | 0.9588 |
| 8 | 0.9731 | 0.8434 |
| 15 | 0.9075 | 0.5374 |

Training substantially improved long-range age discrimination, but adjacent
ages remain almost indistinguishable. This is a low-frequency smooth recency
code, not a sharply orthogonal sixteen-position frequency code.

Magnitude-weighted query/key coefficient phase coherence falls from `0.369` at
H1 to `0.319` at H4, `0.162` at H8, and `0.093` at H16. Thus the accumulated
scalar coefficient phases become increasingly dispersed. There is no global
stationary-phase lock across channel-write terms at late horizons.

## 2. Constructive and destructive interference

The channel expansion reconstructs every model correction with maximum absolute
error `1.79e-7`, well inside the registered `1e-5` criterion. Signed projection
masses onto the actual hidden correction are:

| Read horizon | Positive mass | Negative mass | Negative/positive | Gross mass |
|---:|---:|---:|---:|---:|
| H1 | 1.001 | 0.001 | 0.001 | 1.002 |
| H2 | 1.004 | 0.004 | 0.004 | 1.009 |
| H4 | 1.028 | 0.028 | 0.027 | 1.056 |
| H8 | 1.373 | 0.373 | 0.269 | 1.746 |
| H16 | 1.793 | 0.793 | 0.433 | 2.586 |

Real destructive interference therefore exists and becomes substantial with
depth. At H16, negative terms cancel about 44% of the positive projected mass
(30.7% of gross absolute mass), leaving the exact unit net projection.

This cancellation is not selective retrieval. At H16:

- 2,048 `(write,head,key)` terms are available;
- the mean effective absolute-projection count is `488.6`;
- the largest single term carries only `0.71%` of gross absolute projection;
- the current write carries `8.35%` of gross absolute projection;
- the oldest-to-newest net write contributions increase smoothly from `4.56%`
  to `11.25%`, rather than forming an isolated age peak.

The effective count based on raw contribution norms is even broader, `566.5`.
Consequently the model has learned genuine phase-mediated cancellation plus a
smooth recency bias, but not the proposed mechanism in which a small relevant
set becomes coherently reinforced while unrelated writes disappear through
destructive interference.

The scalar coefficient cosine and hidden-space sign are deliberately kept
separate in the TSV. Since `v` and the output projection are complex-valued,
`cos(arg(conj(q)Uk))` alone is not the sign of a semantic hidden contribution.

