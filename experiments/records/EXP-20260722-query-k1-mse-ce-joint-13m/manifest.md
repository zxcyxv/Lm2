# EXP-20260722: Query K=1, joint stop-gradient state MSE + token CE, 13M

## Status and provenance

- State: preregistered before execution
- Authorization: user-requested joint MSE+CE objective, one-step (K=1) only
- Producer: `train_query_k1_mse_ce_joint_13m.py`
- Seed: 1337; scratch initialization
- Data: WikiText-103 train for optimization; a freshly sampled 128-start fixed
  validation set (seed 1337+999, same convention as the source comparator);
  test split unread
- Comparator: `EXP-20260721-query-k1-inverse-head-ablation-13m/query_inverse_rms`
  (identical architecture, pure teacher-forced CE only), best validation NLL
  `4.4402854442596436`, accuracy `0.2591552734375` at step 1000

## Question

Every MSE-based objective tried in this project so far (I-JEPA EMA state MSE,
decoder-pullback multihorizon MSE, conditional/marginal flow NLL) used that
objective alone. Every CE-based K=1 experiment used CE alone. None combined
MSE and CE into one joint loss. Does adding a stop-gradient state-matching MSE
term to the ordinary K=1 teacher-forced CE change one-step next-token quality
relative to the CE-only comparator, at equal architecture, data, steps, and
schedule?

## Objective

For every causal anchor `t` in a dense 256-position window (identical dense
per-position supervision to the CE-only comparator):

~~~text
q_(t+1)      = contextual query encoder state (same architecture as comparator)
s_(t+1)      = K q_(t+1)
y_(t+1)      = exact_inverse_decode([h_<=t, s_(t+1)])
logits       = rms_tied_head(y_(t+1))
gold_(t+1)   = stop_gradient(encoder([x_0..x_t, x_(t+1)]))_(t+1)   # SAME online
               encoder, no separate EMA copy
CE           = mean cross_entropy(logits, x_(t+1))
MSE          = mean_t || s_(t+1) - gold_(t+1) ||^2 / ||gold_(t+1)||^2   # relative
               (target-norm-normalized) MSE, matching this project's existing
               gold_state_relative_mse diagnostic
loss         = CE + MSE                                            # equal
               weight; not tuned
~~~

The stop-gradient target reuses the same online encoder weights (no EMA copy),
which removes the previously-flagged EMA/online-coordinate mismatch but means
target and prediction share parameters and gradients every step (the target
is only frozen for the current backward pass, not literally lagged in time).
`gold_(t+1)` is exactly the diagnostic already computed as
`gold_state_relative_mse` in the comparator's own `evaluate()`; this
experiment is the first to put it into the training loss instead of only
measuring it.

## Hypothesis under test

The JEPA retirement decision
([`retire-jepa-hidden-state-mse.md`](../../wiki/decisions/retire-jepa-hidden-state-mse.md))
found that MSE-only training on a possibly multi-modal branch state collapses
toward the conditional mean and decouples from decoded token quality. Adding
CE in the same loss could anchor the prediction toward the actually-realized
token's distribution and counteract that collapse, or the two terms' optima
could conflict and produce no net benefit or a worse fit than CE alone. This
experiment does not resolve which; it only measures the one-step outcome
under equal weighting.

## Data and optimization

Identical to the CE-only comparator: vocabulary 8192, width 896, 2 reversible
encoder blocks, `rms-tied` head, context 256, batch 128, 1000 AdamW steps,
peak LR `3e-4`, 100-step warmup, cosine decay, gradient clip 1.0, strict FP32
with TF32 disabled, seed 1337.

## Success and stopping

Preflight must show finite logits, finite nonzero CE gradient, finite nonzero
MSE gradient into both `K` and the encoder, and the exact-inverse roundtrip
gate (relative L2 `< 1e-4`), before any optimizer step. All 1000 steps must
complete on the same fixed schedule as the comparator. Checkpoint selection
uses lowest validation NLL among preregistered report steps; the joint loss
value is not used for selection so that comparison against the CE-only NLL
stays apples-to-apples. A single run, single seed: this cannot establish
whether joint MSE+CE is better in general, only whether it helps or hurts in
this one matched setting.

## Aborted first attempt (raw, unnormalized MSE)

A first run used raw (unnormalized) squared L2 for the MSE term, matching the
retired `k1_shared_stopgrad_state_loss` precedent's literal form. It was
killed manually at step 250 after visible divergence: `K` gradient norm grew
`51 -> 97 -> 1,422 -> 15,500` over steps `1/50/100/250`, raw train MSE grew
`16.1 -> 2.2 -> 15.5 -> 57.3`, and validation NLL *worsened* after step 100
(`7.63 -> 7.35 -> 8.42`) while accuracy stayed near `0.02-0.05`, far below the
CE-only comparator's trajectory at the same steps. Over the same steps
`gold_state_relative_mse` fell to near zero (`0.0001`) while raw MSE grew,
indicating the encoder's own hidden-state norms were inflating rather than
directions aligning -- the raw squared-L2 term rewards growing the predicted
vector's norm, since it is not normalized against anything. This run's
checkpoints and log are not used as evidence for the objective itself, only
as the reason the loss definition was corrected to relative MSE above.
