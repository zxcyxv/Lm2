# Interpretation

The preregistered initial mechanism criterion did not pass.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
The closure target was the gradient-free EMA encode-and-immediate-redecode distribution of P's own greedy token. Znext was decoded through the frozen EMA inverse, which preserved gradients only to the online Znext input.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 6.336482 | 2.347617 |
| H1 latent relative MSE | 1.000298 | 1.000925 |
| H1 posterior target accuracy | 0.002197 | 0.321777 |
| Mean continuous/AR state cosine | -0.001089 | 0.001981 |
| H2--H4 token agreement | 0.644450 | 0.281982 |
| No-write H2--H4 agreement | 0.638916 | 0.282227 |
| Posterior/readout mean logit delta | 0.353412 | 0.234340 |

## EMA self-redecode closure

| Metric | Step 0 | Final |
|---|---:|---:|
| Canonical-to-Znext KL | 2.134088 | 0.715272 |
| Canonical-to-P KL | 1.959817 | 1.254993 |
| Znext/P selected top-1 | 0.632812 | 0.817871 |
| Znext selected probability | 0.134384 | 0.550962 |
| Znext max probability | 0.155383 | 0.571202 |
| Znext max probability >= 0.99 | 0.000000 | 0.036865 |
| Znext entropy | 4.027079 | 1.445548 |
| Znext/canonical relative MSE (diagnostic) | 1.000406 | 0.996537 |
