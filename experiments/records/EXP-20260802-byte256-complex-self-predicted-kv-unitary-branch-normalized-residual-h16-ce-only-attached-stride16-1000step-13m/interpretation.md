# Interpretation

The preregistered initial mechanism criterion did not pass.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
Raw latent and memory carriers; fixed non-affine RMS only on QKV and decoder coupling branches; actual per-instance memory Frobenius measurement normalization; one ungated output correction; fully attached H16 CE; no count schedule, detach, carrier norm, or damping.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 2.460952 | 2.049449 |
| H1 latent relative MSE | 0.125563 | 0.069227 |
| H1 posterior target accuracy | 0.012695 | 0.011719 |
| Mean continuous/AR state cosine | 0.969955 | 0.975566 |
| H2--H4 token agreement | 0.257227 | 0.268229 |
| No-write H2--H4 agreement | 0.185091 | 0.192969 |
| Posterior/readout mean logit delta | 3.681954 | 5.592535 |

## Branch-normalized unitary recurrence

Raw latent and memory carriers used unitary transport. Fixed RMS was applied only before Q/K/V, the measurement was divided by the actual memory Frobenius norm, and the ungated measurement correction was added to the raw latent carrier. Only the decoder branch RMS-normalized the successor.

Final raw pre-clip gradient norm: 0.995678.

## Exact continuation result

The resumed raw gradient norm was `2.3223 / 1.5000 / 1.2108 / 1.0437 /
1.4336 / 0.7848 / 0.9957` at steps 400--1000. No late spike above 10
occurred. Block NLL improved from the resumed step-300 baseline `3.149194` to
`3.054808`, and every H1--H16 NLL improved.

The generated summary's overall false flag is a reporting-scope artifact:
the inherited checker requests step-50/100/200 rows, which a record beginning
at resumed step 300 intentionally does not contain. It is not a non-finite or
late-stability failure in the continuation.
