# H16 scale and Jacobian audit

The tables below are descriptive checkpoint evidence. Causal claims remain limited to the registered frozen-weight counterfactuals.

| Condition | Step | Raw block NLL | Logit RMS |
|---|---:|---:|---:|
| unnormalized_init | 0 | 6.518353 | 1.638199 |
| unnormalized_step0100 | 100 | 4.026980 | 5.678187 |
| unnormalized_step0300 | 300 | 6.119407 | 20.586609 |
| intermediate_init | 0 | 46.675812 | 16.099056 |
| intermediate_step0100 | 100 | 3.604686 | 2.809408 |
| intermediate_step0500 | 500 | 3.237291 | 3.714939 |
| intermediate_step1000 | 1000 | 3.270528 | 3.922625 |
| boundary_init | 0 | 6.493996 | 1.630431 |
| boundary_step0100 | 100 | 6.700233 | 10.313858 |
| boundary_step0300 | 300 | 223.264603 | 104.196495 |

Detailed activation scales, matrix norms, tangent gains, and readout counterfactuals are stored in the adjacent TSV files.
