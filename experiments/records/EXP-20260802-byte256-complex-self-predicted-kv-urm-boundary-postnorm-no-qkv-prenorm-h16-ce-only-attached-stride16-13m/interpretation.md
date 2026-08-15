# Interpretation

The preregistered initial mechanism criterion did not pass.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
The registered final-carrier-only H16 transition was retained exactly, except Q/K/V now receive their raw hidden inputs without the previous shared learned RMSNorm. The initial carrier remains raw, P and all internal innovation/read computations remain unnormalized, and fixed non-affine normalization is applied only to the complete successor Z/S carrier. The fully attached objective is sixteen equal token CEs with no auxiliary target, count scaling, detach, EMA, or token feedback.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 6.521602 | 2.730535 |
| H1 latent relative MSE | 1.000004 | 2.630563 |
| H1 posterior target accuracy | 0.005859 | 0.179688 |
| Mean continuous/AR state cosine | -0.001525 | -0.020550 |
| H2--H4 token agreement | 0.582357 | 0.836784 |
| No-write H2--H4 agreement | 0.582357 | 0.304036 |
| Posterior/readout mean logit delta | 0.273726 | 4.161907 |

## Q/K/V pre-normalization removal

This run changed only the shared learned RMSNorm immediately before the central Q/K/V projections. All boundary-postnorm computation, including the raw initial carrier and raw decoder-facing P, remained unchanged.

## Final-carrier-only boundary post-normalization

P, innovation, and the raw successor memory/read remained unnormalized. Fixed non-affine norms were applied only to the complete successor hidden and memory fields immediately before they were reused by the next central step.

| Metric | Boundary-only | Intermediate-norm control |
|---|---:|---:|
| Final block NLL | 3.167410 | 3.136612 |
| Step-50 raw global gradient norm | 42.898861 | 765.149048 |
| Step-100 raw global gradient norm | 116.907516 | 236.737488 |

## Attached stride-sixteen H16 CE

Sixteen stride-16 anchors and sixteen horizons retained 256 CE labels per sequence while increasing central sequential depth.

| Horizon | Initial NLL | Final NLL | Final target accuracy |
|---:|---:|---:|---:|
| H1 | 6.521602 | 2.730535 | 0.272461 |
| H2 | 6.436055 | 3.223295 | 0.178711 |
| H3 | 6.402783 | 3.155676 | 0.198242 |
| H4 | 6.467681 | 3.136142 | 0.202148 |
| H5 | 6.480776 | 3.279897 | 0.174805 |
| H6 | 6.359838 | 3.217207 | 0.187500 |
| H7 | 6.420347 | 3.230782 | 0.175781 |
| H8 | 6.353940 | 3.172193 | 0.171875 |
| H9 | 6.258066 | 3.196223 | 0.173828 |
| H10 | 6.324080 | 3.184398 | 0.183594 |
| H11 | 6.281308 | 3.150840 | 0.198242 |
| H12 | 6.325443 | 3.186726 | 0.198242 |
| H13 | 6.326617 | 3.213670 | 0.208008 |
| H14 | 6.359868 | 3.226231 | 0.168945 |
| H15 | 6.383408 | 3.226571 | 0.176758 |
| H16 | 6.507006 | 3.148177 | 0.192383 |

H16 block NLL: 6.388051 -> 3.167410.
H4/stride-4 block NLL: 2.385889.
H16 peak VRAM: 2827057664 bytes; H4/stride-4 peak: 2816060416 bytes.
