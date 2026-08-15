# Interpretation

The preregistered initial mechanism criterion did not pass.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
Raw latent and memory carriers; fixed non-affine RMS only on QKV and decoder coupling branches; actual per-instance memory Frobenius measurement normalization; one ungated output correction; fully attached H16 CE; no count schedule, detach, carrier norm, or damping.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 2.049449 | 1.536380 |
| H1 latent relative MSE | 0.069227 | 0.052860 |
| H1 posterior target accuracy | 0.011719 | 0.009766 |
| Mean continuous/AR state cosine | 0.975566 | 0.966793 |
| H2--H4 token agreement | 0.268229 | 0.296029 |
| No-write H2--H4 agreement | 0.192969 | 0.048112 |
| Posterior/readout mean logit delta | 5.592535 | 5.799765 |

## Branch-normalized unitary recurrence

Raw latent and memory carriers used unitary transport. Fixed RMS was applied only before Q/K/V, the measurement was divided by the actual memory Frobenius norm, and the ungated measurement correction was added to the raw latent carrier. Only the decoder branch RMS-normalized the successor.

Final raw pre-clip gradient norm: 0.537117.

## Continuation outcome

The inherited summary marks the overall mechanism criterion false because this
continuation record does not contain steps 50, 100, 200, or 300 and therefore
cannot satisfy the producer's early-step lookup checks. That is a record-scope
artifact, not a late-training instability: from steps 1001--6000 all registered
gradient norms were finite and below 2.10. Block validation NLL improved from
3.054808 at resume to 2.908302, while H1 NLL improved from 2.049449 to 1.536380.
