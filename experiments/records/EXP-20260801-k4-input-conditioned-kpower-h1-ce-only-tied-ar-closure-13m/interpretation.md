# Interpretation

> Scope correction: this report used a matrix frozen at the initial boundary.
> It does not evaluate the intended state-dependent recursion
> `u_(j+1) = K(u_j) u_j`. Its numeric evidence is retained only as a
> block-frozen ablation; the byte-follow-up verdict is superseded.

The preregistered greedy-AR closure-improvement criterion failed; the byte-level follow-up is required.

The state reference is the model's own greedy AR generation followed by canonical re-encoding. No teacher-forced future hidden state is used in these closure metrics.

| Metric | Step 0 | Final |
|---|---:|---:|
| Median state cosine | 0.986610 | 0.546978 |
| Median state relative MSE | 0.026741 | 0.810769 |
| H2--H4 token agreement | 0.998454 | 0.054769 |
| H1 validation NLL | 11.943760 | 4.408783 |

Preregistered criterion components:

- cosine_horizons_improved: 0.0
- relative_mse_horizons_improved: 0.0
- late_reports_above_initial_median_cosine: 0.0
- final_median_cosine_gain: -0.43963184246786113
