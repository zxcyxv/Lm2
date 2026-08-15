# Interpretation

The preregistered initial mechanism criterion did not pass.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
The matched initial-root-RMS H16 recurrence was retained except its successor hidden is the fixed-RMS-normalized sum of the unitary rotated previous hidden and the innovation-only read. P, the KV write, complete memory update, CE, optimizer, and all gradient paths remain unchanged.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 6.529102 | 2.585202 |
| H1 latent relative MSE | 7.160352 | 0.492376 |
| H1 posterior target accuracy | 0.012695 | 0.012695 |
| Mean continuous/AR state cosine | 0.497014 | 0.807911 |
| H2--H4 token agreement | 0.431771 | 0.959635 |
| No-write H2--H4 agreement | 0.431836 | 0.949870 |
| Posterior/readout mean logit delta | 4.341024 | 3.963331 |

## Rotated-hidden innovation residual

The sole transition change was `Znext = fixed_RMS(RZ + innovation_delta)`. The CE-facing P, rank-one write, complete memory update, initial-root normalization, and final Z/S boundary normalization were retained.

| Step | Candidate gnorm | Matched gnorm | Candidate block NLL | Matched block NLL |
|---:|---:|---:|---:|---:|
| 1 | 235.902512 | 215.868988 | 6.499490 | 6.129311 |
| 50 | 53.555164 | 21.100031 | 3.509794 | 3.353405 |
| 100 | 35.028534 | 68.031082 | 3.453836 | 3.558278 |
| 200 | 16.550217 | 66.590050 | 3.182449 | 3.175779 |
| 300 | 33.470047 | 49.031799 | 3.203612 | 3.163019 |