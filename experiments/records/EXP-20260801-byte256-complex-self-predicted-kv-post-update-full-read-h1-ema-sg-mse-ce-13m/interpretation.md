# Interpretation

The preregistered initial mechanism criterion did not pass.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
The H1 target came from a gradient-free full-model EMA, and all reported primary inference metrics used that same EMA snapshot.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 6.336482 | 1.394581 |
| H1 latent relative MSE | 1.000298 | 0.335577 |
| H1 posterior target accuracy | 0.002197 | 0.338867 |
| Mean continuous/AR state cosine | -0.001089 | 0.460377 |
| H2--H4 token agreement | 0.644450 | 0.242269 |
| No-write H2--H4 agreement | 0.638916 | 0.177897 |
| Posterior/readout mean logit delta | 0.353412 | 5.212979 |

## Agreement monitoring

H2--H4 agreement from steps 200 through 1000 was:

`0.193522, 0.220703, 0.246501, 0.240560, 0.247070, 0.247152, 0.242350, 0.242676, 0.242269`.

This is not monotonic. H2 improved at every registered point, but H3/H4 and
exact-block agreement did not, so the deeper horizons erased the H2 gain.

The residual/Shapley audit found that the full-read recurrence is better than
the old formula at identical weights, but its innovation changes from strongly
helpful at H1 to directionally harmful at H3/H4. See the sibling audit record
for the exact decomposition.
