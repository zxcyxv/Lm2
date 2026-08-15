# Stopped online-SG interpretation

The user stopped this run after the step-250 report to replace the moving
online target with an EMA target and to use a matched EMA encoder/inverse
decoder at inference. No step-1000 verdict is assigned.

| Metric | Step 0 | Step 100 | Step 250 |
|---|---:|---:|---:|
| H1 validation NLL | 6.336482 | 2.390366 | 1.745664 |
| H1 latent relative MSE | 0.999923 | 0.592302 | 0.455643 |
| H1 posterior target accuracy | 0.001953 | 0.257080 | 0.288330 |
| Mean central/AR state cosine | 0.000907 | 0.595454 | 0.433149 |
| H2--H4 token agreement | 0.645101 | 0.361410 | 0.253743 |
| No-write H2--H4 agreement | 0.641439 | 0.425293 | 0.157715 |

The stop-gradient target improved H1 token and posterior-target metrics, but
the central/AR state cosine and H2--H4 agreement were already falling by step
250. This is scoped evidence motivating the EMA ablation, not evidence about
its eventual step-1000 behavior.
