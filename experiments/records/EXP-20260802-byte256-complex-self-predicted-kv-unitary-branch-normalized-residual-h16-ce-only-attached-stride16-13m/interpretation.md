# Interpretation

The preregistered initial mechanism criterion did not pass.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
Raw latent and memory carriers; fixed non-affine RMS only on QKV and decoder coupling branches; actual per-instance memory Frobenius measurement normalization; one ungated output correction; fully attached H16 CE; no count schedule, detach, carrier norm, or damping.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 48.574527 | 2.460952 |
| H1 latent relative MSE | 0.965087 | 0.125563 |
| H1 posterior target accuracy | 0.012695 | 0.012695 |
| Mean continuous/AR state cosine | 0.975603 | 0.969955 |
| H2--H4 token agreement | 0.997266 | 0.257227 |
| No-write H2--H4 agreement | 1.000000 | 0.185091 |
| Posterior/readout mean logit delta | 3.327732 | 3.681954 |

## Branch-normalized unitary recurrence

Raw latent and memory carriers used unitary transport. Fixed RMS was applied only before Q/K/V, the measurement was divided by the actual memory Frobenius norm, and the ungated measurement correction was added to the raw latent carrier. Only the decoder branch RMS-normalized the successor.

Final raw pre-clip gradient norm: 2.206702.