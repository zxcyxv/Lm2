# Interpretation

The preregistered initial mechanism criterion did not pass.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
Equal H1--H4 token CE on a state-dependent recurrence. At every central block boundary only the previous hidden autograd edge was stopped; the complex memory carrier remained attached. Q/K/V consumed the detached post-normalized hidden directly, memory used unitary transport plus an unmodified rank-one write, the read was raw q^dagger S/sqrt(key_dim), and fixed non-affine RMS of hidden-plus-readout formed the actual next state. No memory normalization, damping, EMA, or auxiliary loss.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 46.449434 | 4.624183 |
| H1 latent relative MSE | 7.242508 | 1.853097 |
| H1 posterior target accuracy | 0.012695 | 0.130859 |
| Mean continuous/AR state cosine | 0.203739 | 0.077090 |
| H2--H4 token agreement | 0.095378 | 0.028255 |
| No-write H2--H4 agreement | 0.990690 | 0.074479 |
| Posterior/readout mean logit delta | 0.000000 | 0.000000 |

## Step-100 gradient-localization precursor

This run preserved the detached-hidden, raw-read, recurrent-postnorm architecture and the 33,570-step learning-rate schedule, but stopped after producing the fresh step-100 checkpoint used by the separate gradient-localization audit.

Step-100 raw pre-clip global gradient norm: 102.742355.