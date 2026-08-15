# Interpretation

The preregistered initial mechanism criterion did not pass.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
Only hidden inputs to central Q/K/V were pre-RMS-normalized. Raw memory used sqrt(write_count)-scaled accumulated reads, and the raw hidden successor was the unitary rotated carrier plus innovation-only delta. No initial-root or recurrent post-normalization was present.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 6.529102 | 3.145767 |
| H1 latent relative MSE | 0.964672 | 1.942443 |
| H1 posterior target accuracy | 0.012695 | 0.009766 |
| Mean continuous/AR state cosine | 0.495431 | 0.426939 |
| H2--H4 token agreement | 0.415169 | 0.391732 |
| No-write H2--H4 agreement | 0.370703 | 0.374609 |
| Posterior/readout mean logit delta | 1.651816 | 20.275646 |

## Input-pre-norm raw hidden residual

Only real hidden inputs to the shared Q/K/V projections were RMS-normalized. Memory remained raw and accumulated-memory reads used `1/sqrt(write_count)`. The raw successor was `RZ + innovation_delta` with no recurrent post-normalization.

Final raw pre-clip gradient norm: 17.269508.