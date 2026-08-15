# Interpretation

The preregistered initial mechanism criterion passed.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
The registered 121M LR-2e-4 feedback recurrence with only the hidden successor addition removed: z_next equals the Frobenius-normalized measurement delta rather than rotated_z plus that delta.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 55.559313 | 36.747838 |
| H1 latent relative MSE | 1.000034 | 1.025668 |
| H1 posterior target accuracy | 0.001953 | 0.048828 |
| Mean continuous/AR state cosine | -0.000042 | -0.144782 |
| H2--H4 token agreement | 0.007617 | 0.654687 |
| No-write H2--H4 agreement | 0.005794 | 0.002344 |
| Posterior/readout mean logit delta | 12.733054 | 4.567684 |

## H4-supervised feedback recurrence with H16 monitoring

Training unfolded the state-dependent feedback recurrence through H4 and backpropagated equal CE from H1--H4 only. Validation unfolded the same shared transition through H16; H5--H16 therefore measure out-of-objective rollout transfer.

| Horizon | Supervised in training | Initial NLL | Final NLL |
|---:|:---:|---:|---:|
| H1 | yes | 55.559312 | 36.747837 |
| H2 | yes | 52.817238 | 36.768068 |
| H3 | yes | 52.266114 | 38.514548 |
| H4 | yes | 49.375423 | 37.391207 |
| H5 | no | 48.129832 | 37.783351 |
| H6 | no | 49.850965 | 36.955019 |
| H7 | no | 51.042783 | 38.519449 |
| H8 | no | 49.980556 | 38.423123 |
| H9 | no | 49.685404 | 37.087121 |
| H10 | no | 49.384639 | 38.726178 |
| H11 | no | 49.311592 | 37.134646 |
| H12 | no | 49.075969 | 37.993305 |
| H13 | no | 49.986707 | 38.151701 |
| H14 | no | 48.969758 | 39.076504 |
| H15 | no | 49.278142 | 38.795277 |
| H16 | no | 49.855004 | 38.502006 |