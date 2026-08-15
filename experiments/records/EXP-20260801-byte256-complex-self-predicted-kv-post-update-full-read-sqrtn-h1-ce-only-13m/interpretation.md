# Interpretation

The preregistered initial mechanism criterion passed.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
The sqrt(n)-normalized post-update full-read recurrence executed one target-free step. Its innovation-free P state received corpus H1 CE only; no hidden, KL, EMA, or token-reencoding target was present.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 6.336482 | 1.879377 |
| H1 latent relative MSE | 1.000158 | 1.002934 |
| H1 posterior target accuracy | 0.002197 | 0.180420 |
| Mean continuous/AR state cosine | -0.000990 | 0.002148 |
| H2--H4 token agreement | 0.658610 | 0.202555 |
| No-write H2--H4 agreement | 0.640951 | 0.203776 |
| Posterior/readout mean logit delta | 0.333718 | 1.214967 |

## Matched H1 CE-only control

H1 validation NLL: 6.336482 -> 1.879377.

The matched four-horizon CE run reached H1 NLL 2.168469 at step 200. However,
H1-only block NLL was 3.487789 versus 2.769732 for four-horizon CE. H1-only
therefore optimized the immediate token faster while failing to train the
later latent rollout. The older MSE+CE predecessor is not used for this direct
conclusion because its objective and normalization differ.
