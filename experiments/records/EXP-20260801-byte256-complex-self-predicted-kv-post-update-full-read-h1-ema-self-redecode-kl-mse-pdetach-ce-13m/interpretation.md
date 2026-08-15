# Interpretation

The preregistered initial mechanism criterion passed.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
The self-selected EMA canonical hidden and its immediate inverse-decode distribution jointly supervised Znext. P stayed attached to ordinary CE but was detached where it entered successor K/V/Q projections. Pre-update memory remained attached.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 6.336482 | 2.215206 |
| H1 latent relative MSE | 1.000298 | 0.878796 |
| H1 posterior target accuracy | 0.002197 | 0.357666 |
| Mean continuous/AR state cosine | -0.001089 | 0.440373 |
| H2--H4 token agreement | 0.644450 | 0.382568 |
| No-write H2--H4 agreement | 0.638916 | 0.375814 |
| Posterior/readout mean logit delta | 0.353412 | 0.641944 |

## P-detached joint closure

The CE-visible P remained attached to ordinary token CE. Both EMA hidden MSE and EMA decode KL used the same successor graph, whose K(P), V(P), and Q(P) inputs treated P as a fixed value.

| Metric | Step 0 | Final | KL-only final |
|---|---:|---:|---:|
| H1 canonical hidden relative MSE | 1.000406 | 0.768830 | 0.996537 |
| H1 canonical hidden cosine | -0.003862 | 0.503774 | 0.061611 |
| H1 canonical-to-Znext KL | 2.134088 | 0.588270 | 0.715272 |
| H1 P/Znext top-1 | 0.632812 | 0.854248 | 0.817871 |
| H1 validation NLL | 6.336482 | 2.215206 | 2.347617 |
