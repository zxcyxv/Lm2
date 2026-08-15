# Interpretation

The preregistered initial mechanism criterion did not pass.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
Raw latent and memory carriers; fixed non-affine RMS only on QKV and decoder coupling branches; actual per-instance memory Frobenius measurement normalization; one ungated output correction; fully attached H16 CE; no count schedule, detach, carrier norm, or damping.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 48.077450 | 3.702009 |
| H1 latent relative MSE | 0.913746 | 0.066262 |
| H1 posterior target accuracy | 0.062500 | 0.062500 |
| Mean continuous/AR state cosine | 0.973584 | 0.969682 |
| H2--H4 token agreement | 1.000000 | 0.512500 |
| No-write H2--H4 agreement | 1.000000 | 0.808333 |
| Posterior/readout mean logit delta | 3.281217 | 2.794403 |

## Branch-normalized unitary recurrence

Raw latent and memory carriers used unitary transport. Fixed RMS was applied only before Q/K/V, the measurement was divided by the actual memory Frobenius norm, and the ungated measurement correction was added to the raw latent carrier. Only the decoder branch RMS-normalized the successor.

Final raw pre-clip gradient norm: 29.384308.