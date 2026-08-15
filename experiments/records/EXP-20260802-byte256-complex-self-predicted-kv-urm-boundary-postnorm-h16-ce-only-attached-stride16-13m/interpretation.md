# Interpretation

The final-carrier-only boundary normalization hypothesis failed at H16 under
the registered optimizer and schedule. The run was stopped after the step-300
evaluation because continuing no longer provided a proportionate stability
test.

The implementation did satisfy the intended placement: P, innovation, and the
complete raw successor were formed without post-normalization; only the final
hidden and memory fields were normalized before reuse. Future-input
independence and causal decoding passed preflight.

| Step | Block NLL | Raw global grad norm | H1 P RMS | H1 innovation energy | H1 logit RMS |
|---:|---:|---:|---:|---:|---:|
| 0 | 6.385139 | n/a | 0.003871 | 0.390732 | 1.733564 |
| 50 | 4.847892 | 1234.796387 | 0.133776 | 3.373547 | 4.052115 |
| 100 | 6.448073 | 477.393280 | 0.987765 | 26.041762 | 13.148078 |
| 200 | 3.295818 | 56.261436 | 0.991152 | 97.113506 | 6.425661 |
| 300 | 223.148930 | 604649216.000000 | 28.007025 | 136.381794 | 373.724863 |

Throughout these reports, recurrent hidden RMS and recurrent memory RMS
remained approximately one at every horizon. Boundary normalization therefore
controlled the stored carrier magnitude but did not control the raw transition
computed inside the block. Raw P, innovation energy, logits, and the backward
Jacobian remained free to grow.

The result conflicts with the expectation that moving every norm to the final
carrier boundary would improve all aspects of stability. It did correct the
initial decoder-scale distortion: initial block NLL was `6.385139`, compared
with `45.232199` for the intermediate-norm control. At step 50 its block NLL
was also slightly lower (`4.847892` versus `4.882202`). Those benefits did not
survive the unconstrained internal trajectory. The intermediate-norm control
completed 1000 steps at block NLL `3.136612` and final gradient norm
`10.558392`.

This failure does not establish that every boundary-normalized transition is
unstable. It is scoped to the present split-output transition, in which the
decoder-facing P is not the normalized recurrent carrier. In URM, the
post-normalized hidden is both the recurrent carrier and the state consumed by
the output head; this architecture does not share that single-state property.

The test split was not materialized or read.
