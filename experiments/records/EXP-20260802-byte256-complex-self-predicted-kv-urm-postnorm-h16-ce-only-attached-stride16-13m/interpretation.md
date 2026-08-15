# Interpretation

The preregistered initial mechanism criterion passed.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
The fully attached H16 central recurrence used fixed non-affine post-RMSNorm for P and recurrent Z, complex RMSNorm for the complete memory carrier, and no count-dependent read scaling. Sixteen P states received equal token CE in one causal decode; no auxiliary target, detach, EMA, or token feedback was present.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 45.908121 | 2.626265 |
| H1 latent relative MSE | 10.055689 | 1.141715 |
| H1 posterior target accuracy | 0.006836 | 0.024414 |
| Mean continuous/AR state cosine | -0.000825 | 0.068474 |
| H2--H4 token agreement | 0.005013 | 0.464062 |
| No-write H2--H4 agreement | 0.006185 | 0.097982 |
| Posterior/readout mean logit delta | 17.921938 | 42.005792 |

## URM-style central post-normalization

Count-dependent read scaling was disabled. P, recurrent Z, and the complete complex memory carrier received fixed non-affine RMS post-normalization after their respective updates.

## Attached stride-sixteen H16 CE

Sixteen stride-16 anchors and sixteen horizons retained 256 CE labels per sequence while increasing central sequential depth.

| Horizon | Initial NLL | Final NLL | Final target accuracy |
|---:|---:|---:|---:|
| H1 | 45.908121 | 2.626265 | 0.260742 |
| H2 | 44.556870 | 3.030093 | 0.185547 |
| H3 | 45.809504 | 3.114225 | 0.195312 |
| H4 | 45.527175 | 3.127085 | 0.205078 |
| H5 | 45.401252 | 3.250829 | 0.173828 |
| H6 | 44.752317 | 3.191837 | 0.186523 |
| H7 | 45.241843 | 3.224218 | 0.173828 |
| H8 | 44.944150 | 3.166682 | 0.172852 |
| H9 | 44.798946 | 3.171537 | 0.173828 |
| H10 | 44.872131 | 3.189108 | 0.184570 |
| H11 | 45.490608 | 3.139920 | 0.200195 |
| H12 | 44.595226 | 3.184251 | 0.198242 |
| H13 | 44.747279 | 3.194481 | 0.208008 |
| H14 | 45.373011 | 3.223030 | 0.169922 |
| H15 | 45.939721 | 3.200401 | 0.176758 |
| H16 | 45.757027 | 3.151831 | 0.192383 |

H16 block NLL: 45.232199 -> 3.136612.
H4/stride-4 block NLL: 2.385889.
H16 peak VRAM: 3043057664 bytes; H4/stride-4 peak: 2816060416 bytes.
