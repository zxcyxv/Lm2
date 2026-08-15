# Interpretation

## Verdict

The preregistered mechanism criterion did not pass. The model learned a good
one-step posterior latent regression and a useful self-predicted innovation,
but it did not reach greedy-AR-equivalent discrete behavior through horizons
2--4.

The decoder received only the innovation-free prior. The full innovation was
used only as the recurrent posterior and latent-regression prediction. The
checkpoint contains no beta parameter, temperature coefficient, observed or
generated token write, EMA state, or separate vocabulary head.

## Main measurements

| Metric | Step 0 | Step 250 | Step 500 | Step 1000 |
|---|---:|---:|---:|---:|
| H1 validation NLL | 6.336482 | 2.103873 | 1.856028 | 1.565493 |
| H1 attached relative MSE | 0.999923 | 0.016715 | 0.013829 | 0.013020 |
| Mean continuous/AR state cosine | 0.000907 | 0.961650 | 0.968594 | 0.945871 |
| H2--H4 token agreement | 0.645101 | 0.362549 | 0.358154 | 0.261149 |
| No-write H2--H4 agreement | 0.641439 | 0.125488 | 0.081462 | 0.073649 |
| Posterior/readout mean logit delta | 0.152432 | 2.417766 | 1.854063 | 3.480556 |

The high agreement at initialization is degenerate: the untrained paths emit
the same small set of bytes. It is not evidence of AR equivalence. The useful
comparison starts after token CE becomes nontrivial. Agreement initially rose
from 0.302 at step 50 to 0.363 at step 250, plateaued at step 500, and then
declined as NLL continued to improve.

At step 1000 the write-enabled recurrence exceeded its functional no-write
ablation by 0.188 absolute H2--H4 agreement (0.261 versus 0.074). Thus the
self-predicted innovation is causally useful, although insufficient for the
registered 0.50 threshold.

## Scope of the failure

Horizon-one parallel and greedy-AR logits remained identical up to a maximum
absolute error of 7.63e-5. Held-out future-byte mutations had zero effect in
preflight. The failure therefore does not come from future-token leakage or a
mismatch at the common first proposal.

The final H1 posterior has cosine 0.995 and relative MSE 0.012 against the
online future state, whereas multi-step relative errors remain large despite
high mean cosine. The discrete disagreement is therefore a self-composition
and decoder-boundary problem, not evidence that the innovation is unused.

Moving online coordinates remain a possible secondary factor, but these
results do not identify them as the dominant cause: H1 regression stayed
small while agreement fell. An EMA target would also live in a lagged encoder
coordinate system that is not exactly inverted by the current decoder, so it
should be tested only as a separately registered ablation after measuring
fixed-prefix encoder drift and mismatched-token logit margins.

The prematurely tempered predecessor remains preserved as conflicting scoped
evidence. It reached lower H1 error at step 250 but only 0.089 H2--H4
agreement; because both its causal readout and residual wiring differ, this
does not isolate beta as the cause.
