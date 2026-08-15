# EXP-20260726 warm-start EMA behavioral closure

## Status

- State: completed all 1000 optimizer updates
- Authorization: user requested EMA closure; cold-start preflight schedule was
  stopped at step 100 after stale-teacher failure
- Primary matched control:
  `../EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-open-nll-transition-only-closure-13m/`
- Failed cold-start EMA diagnostic:
  `../EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-open-nll-ema-teacher-transition-only-closure-13m/`
- Original no-Open-CE control:
  `../EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-nll-closure-13m/`
- Test split remains unread
- Metrics: `metrics.tsv`
- Interpretation: `interpretation.md`

### Execution result

- Exact-copy warm start reproduced the matched non-EMA run through step 100.
- At step 500, versus the matched non-EMA run, current-online closure improved
  from `1.480903` to `1.416095`; H2--H4 Open NLL changed from `6.612508`
  to `6.636425`.
- At step 1000, versus the original control, current-online closure improved
  from `1.882352` to `1.797062` and H2--H4 Open NLL improved from `6.456272`
  to `6.445196`.
- Overall Open NLL was slightly worse, `5.959662` to `5.966937`, and
  current-online H2--H4 top-1 agreement changed from `0.236715` to
  `0.232920`.
- The result is therefore partial support for closure stabilization, not
  evidence that EMA resolves the growing Open/canonical gap.

## Question

Can a full-model EMA closure teacher stabilize the recurrent behavioral target
without carrying a large random-initialization component through the period
in which the canonical teacher is being formed?

The cold-start EMA with decay `0.99` retained `36.6%` of initialization at
step 100. Its EMA teacher/Open top-1 agreement was only `0.007884`, and it
degraded Open NLL. This run warm-starts the same EMA before applying the same
long-term decay.

## Fixed architecture and online objectives

The online architecture is unchanged:

`T(h) = K(h)h + R(h)`.

- two-stage state-conditioned exact Givens operator
- deterministic state-conditioned innovation
- canonical latent TF input for direct one-step token NLL
- sequential self-fed latent Open rollout
- no token/embedding feedback inside the central recurrence
- no noise, candidates, prior, latent plan, latent MSE, or InfoNCE

Online token objectives are:

`L_TF = mean_{B:E} CE(logits_TF,j, gold_j)`,

`L_OL = mean_{C:E} CE(logits_OL,j, gold_j)`.

## Warm-start EMA teacher

The EMA is a complete gradient-free copy of the encoder, transition,
exact-inverse decoder, tied head, and embedding.

For online optimizer updates 1 through 100, the post-update EMA decay is
zero:

`theta_EMA <- theta_online`.

Thus the teacher used by the next update is at most one online update behind,
and validation after each reported update sees an exact online copy.

Beginning after optimizer update 101:

`theta_EMA <- 0.99 theta_EMA + 0.01 theta_online`.

The closure target before each update is:

`p_EMA,j = softmax(logits_TF,j(theta_EMA))`.

The student is the online Open distribution:

`p_OL,j = softmax(logits_OL,j(theta_online))`.

## Loss and gradient routing

`L_close = mean_{C:E} KL(p_EMA,j || p_OL,j)`.

At update `s`:

`L = L_TF + L_OL + lambda_s L_close`,

`lambda_s = min(1.0, s / 100)`.

- TF and Open NLL update the complete online model.
- EMA closure updates only the online central transition.
- no EMA parameter has a gradient or optimizer state.
- no closure gradient accumulates in the online encoder, inverse decoder,
  head, or embedding.

Both EMA-target closure and current-online canonical closure are reported.
Both EMA/Open and current-online/Open top-1 agreement are reported, so a
stale-target apparent success is visible.

## Fixed training configuration

- WikiText-103 train/validation; BPE vocabulary 8192
- seed 1337; validation-start seed `1337 + 999`
- identical 128 validation starts to all controls
- width 896; two reversible causal blocks
- exact inverse decoder; RMS-tied vocabulary head
- transition bottleneck 128; two complete Givens stages
- horizons 4; stride-one anchors 0 through 255
- effective batch 64; physical microbatch 16; four accumulations
- strict float32; TF32 disabled
- AdamW; clip norm 1.0
- 1000 updates on the unchanged 6000-update LR schedule
- closure weight ramp over updates 1--100
- EMA exact-copy warm start through update 100
- EMA decay `0.99` beginning at update 101
- reports at 1, 50, 100, 250, 500, 750, 1000
- online and EMA checkpoints at 100, 250, 500, 750, 1000
- test split remains unread

## Structural preflight criteria

- online and EMA parameters are exactly equal before update 1
- decay zero copies all parameters and buffers exactly
- decay `0.99` matches the registered affine update
- EMA closure has finite nonzero gradients in online transition parameters
- routed closure produces no online non-transition gradient
- all EMA parameters remain gradient-free
- H1 is excluded from Open NLL and closure
- online self-fed states/logits have no future-gold dependency
- both online and EMA closure diagnostics are finite

## Success and failure criteria

At step 100, versus the non-EMA matched control:

- EMA/Open H2--H4 agreement differs from current-online/Open agreement by
  less than `0.01`
- Open NLL is no more than `0.03` worse
- H2--H4 Open accuracy is no more than `0.003` worse

At matched steps 250 and 500:

- current-online closure KL is below the non-EMA matched control
- current-online H2--H4 agreement is above the matched control
- H2--H4 Open NLL is no more than `0.05` worse

At step 1000, versus the original control:

- overall Open NLL below `5.959662`
- H2--H4 Open NLL below `6.456272`
- current-online H2--H4 agreement at least `0.286715`
- teacher NLL at most `4.611288`

The EMA hypothesis fails if only EMA-target metrics improve while current
online closure or Open token quality degrades beyond these tolerances.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_k4_ha_skew_state_conditioned_givens_latent_tf_nll_closure_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --experiment-id \
    EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-open-nll-ema-warmstart100-transition-only-closure-13m \
  --open-nll-weight 1.0 \
  --closure-weight 1.0 \
  --closure-warmup-steps 100 \
  --closure-transition-only \
  --closure-teacher-ema-decay 0.99 \
  --closure-teacher-ema-warm-start-steps 100 \
  --record-dir \
    experiments/records/EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-open-nll-ema-warmstart100-transition-only-closure-13m \
  --output-dir \
    outputs/experiments/EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-open-nll-ema-warmstart100-transition-only-closure-13m
```

## Evidence boundary

EMA target stabilization does not add stochastic sampling or prove exact
autoregressive joint-distribution equivalence. Metrics remain TSV and
post-run interpretation remains a separate Markdown artifact.
