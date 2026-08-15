# Interpretation

The preregistered initial mechanism criterion passed.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
The fully attached central recurrence executed four target-free steps at 64 stride-four prefix anchors. Its four P states received equal token CE in one causal decode; no hidden, KL, EMA, detach, or token-reencoding target was present.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 6.336482 | 1.675063 |
| H1 latent relative MSE | 1.000158 | 1.066472 |
| H1 posterior target accuracy | 0.002197 | 0.124512 |
| Mean continuous/AR state cosine | -0.000990 | -0.041317 |
| H2--H4 token agreement | 0.658610 | 0.230387 |
| No-write H2--H4 agreement | 0.640951 | 0.255290 |
| Posterior/readout mean logit delta | 0.333718 | 2.700341 |

## Attached stride-four H4 CE

Only training anchor stride changed: 64 anchors and 256 CE labels per sequence replaced 256 anchors and 1024 overlapping labels.

| Horizon | Initial NLL | Final NLL | Final target accuracy |
|---:|---:|---:|---:|
| H1 | 6.336482 | 1.675063 | 0.503906 |
| H2 | 6.318067 | 2.269957 | 0.363525 |
| H3 | 6.277511 | 2.684619 | 0.281738 |
| H4 | 6.233813 | 2.913915 | 0.235840 |

Stride-four block NLL: 6.291468 -> 2.385889.
Stride-one step-1000 block NLL: 2.245606.
Stride-four peak VRAM: 2816060416 bytes; stride-one peak: 7507823104 bytes.
