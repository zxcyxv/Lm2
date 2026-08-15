# Interpretation

The preregistered initial mechanism criterion did not pass.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
The central recurrence executed four sequential target-free steps. Its four innovation-free P states were decoded as one causal tape and all four corpus-future bytes supplied CE labels only. No hidden, KL, EMA, or token-reencoding target was present.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 6.336482 | 1.539392 |
| H1 latent relative MSE | 1.000158 | 1.089372 |
| H1 posterior target accuracy | 0.002197 | 0.156982 |
| Mean continuous/AR state cosine | -0.000990 | -0.022266 |
| H2--H4 token agreement | 0.658610 | 0.291341 |
| No-write H2--H4 agreement | 0.640951 | 0.320801 |
| Posterior/readout mean logit delta | 0.333718 | 4.165671 |

## Four-token CE block

The central layer executed four sequential continuous transitions. The exact-inverse decoder consumed the four P states as one causal tape and produced all four training logits in one batched call.

| Horizon | Initial NLL | Final NLL | Final target accuracy | Final central/AR agreement |
|---:|---:|---:|---:|---:|
| H1 | 6.336482 | 1.539392 | 0.544922 | 1.000000 |
| H2 | 6.318067 | 2.121224 | 0.396729 | 0.248291 |
| H3 | 6.277511 | 2.521403 | 0.312988 | 0.252441 |
| H4 | 6.233813 | 2.800406 | 0.257324 | 0.373291 |

Four-token mean validation NLL: 6.291468 -> 2.245606.
