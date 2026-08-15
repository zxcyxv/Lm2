# EXP-20260726 state-conditioned open-NLL trajectory training

## Status

- State: stopped by user after the complete step-500 report
- Authorization: user requested the diagnosed objective correction and run
- Primary control:
  `../EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-nll-closure-13m/`
- Test split remains unread

## Question

Does directly supervising the inference-equivalent self-fed latent trajectory
remove the growing teacher/open gap when behavioral closure is retained as a
transition-only distribution regularizer?

The control optimized direct next-token NLL only on canonical latent inputs.
Its open-loop token CE was diagnostic only, while H2--H4 received a
stop-gradient teacher-distribution KL. At step 1000 the control improved
open-loop NLL to `5.959662`, but mean H2--H4 teacher/open top-1 agreement fell
from the step-50 peak `0.580831` to `0.236715`; closure KL rose to `1.882352`.

## Fixed architecture

The architecture is unchanged from the primary control:

`T(h) = K(h)h + R(h)`.

- `K(h)` is the same two-stage state-conditioned exact Givens operator.
- `R(h)` is the same deterministic state-conditioned innovation.
- The central recurrence consumes only latent states.
- There is no token ID or embedding feedback, sampled noise, candidate
  trajectory, trajectory prior, latent plan, latent MSE, or InfoNCE.

Canonical latent teacher forcing supplies observed pre-central states:

`hB_hat_TF = T(hA)`, `hC_hat_TF = T(hB)`, and so on.

The inference-equivalent path remains self-fed:

`hB_tilde = T(hA)`,
`hC_tilde = T(hB_tilde)`,
`hD_tilde = T(hC_tilde)`,
`hE_tilde = T(hD_tilde)`.

No token is decoded and re-embedded inside this recurrence.

## Registered objective and gradient routing

The ordinary canonical-input token objective is unchanged:

`L_TF = mean_{B:E} CE(logits_TF_j, gold_j)`.

The new direct open-loop objective covers only H2--H4 because H1 is the exact
same graph as the canonical H1 prediction:

`L_OL = mean_{C:E} CE(logits_OL_j, gold_j)`.

Behavioral closure remains:

`L_close = mean_{C:E} KL(stopgrad(p_TF_j) || p_OL_j)`.

The registered scalar objective at optimizer update `s` is:

`L = 1.0 * L_TF + 1.0 * L_OL + lambda_s * L_close`,

where

`lambda_s = min(1.0, s / 100)`.

Gradient routing is part of the registration:

- `L_TF + L_OL` updates the complete model.
- `L_close` updates only the shared central transition `T`.
- The KL target remains stop-gradient.
- No KL gradient is accumulated into the encoder, exact-inverse decoder,
  tied vocabulary head, or embedding.

Thus open-loop CE directly optimizes the observed token trajectory without
regressing latent coordinates. KL is no longer the sole open-loop training
signal; it regularizes the recurrent central transition's behavioral
closure.

Teacher and open-loop NLL weights are fixed at `1:1`. KL ramps linearly from
`0.01` at update 1 to `1.0` at update 100, then remains at `1.0`.

This schedule was fixed after the preflight, before update 1. At identity
initialization, open-loop CE and closure transition gradients had cosine
`0.996168` and norms `193.549372` and `194.456574`, respectively, while the
teacher-NLL transition-gradient norm was `63.506847`. Applying full KL weight
from update 1 would therefore duplicate the new open-loop signal before the
teacher becomes useful. Any later change to the registered ramp or final
weight requires a separate ablation.

## Fixed training configuration

- WikiText-103 train and validation; BPE vocabulary 8192
- seed 1337; validation-start seed `1337 + 999`
- identical 128 validation starts as the primary control
- width 896, two reversible causal blocks
- exact inverse decoder and RMS-tied vocabulary head
- transition bottleneck 128 and two complete Givens stages
- four horizons; stride-one anchors 0 through 255
- effective batch 64, physical microbatch 16, four accumulations
- strict float32; TF32 disabled
- AdamW, clip norm 1.0
- KL-weight linear warmup over optimizer updates 1--100
- 1000 optimizer updates on the same 6000-update LR schedule
- reports at 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at 100, 250, 500, 750, and 1000
- test split remains unread

The unchanged LR schedule isolates objective and gradient-routing effects.
The high step-1000 LR remains a known limitation rather than being changed
simultaneously.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_k4_ha_skew_state_conditioned_givens_latent_tf_nll_closure_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --experiment-id \
    EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-open-nll-transition-only-closure-13m \
  --open-nll-weight 1.0 \
  --closure-weight 1.0 \
  --closure-warmup-steps 100 \
  --closure-transition-only \
  --record-dir \
    experiments/records/EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-open-nll-transition-only-closure-13m \
  --output-dir \
    outputs/experiments/EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-open-nll-transition-only-closure-13m
```

## Structural preflight criteria

- the scalar loss exactly equals `L_TF + L_OL + L_close`
- H1 is excluded from `L_OL` and `L_close`
- `L_OL` has finite nonzero gradients to the transition, encoder, and tied
  embedding
- `L_close` has finite nonzero gradients to the transition
- routed `L_close` backward accumulates no gradient in any non-transition
  parameter
- teacher probabilities retain no KL gradient path
- H1 teacher/open states and logits match to float32 tolerance
- future gold token changes do not affect the open latent tape or logits
- conditional Givens norm error remains below `1e-5`

## Step-1000 success and failure criteria

Primary matched-control improvements:

- validation open-loop NLL below control `5.959662`
- mean H2--H4 open-loop NLL below control `6.456272`
- mean H2--H4 teacher/open top-1 agreement at least `0.286715`, five
  percentage points above the control
- behavioral closure KL below control `1.882352`

Supporting:

- validation teacher-forced NLL at most `4.611288`, no more than `0.15`
  above the control
- open-loop accuracy is nonzero at every horizon
- every optimization and validation metric is finite
- learned angles and innovations remain finite and nonzero
- conditional-operator norm error remains below `1e-5`

The run fails its central question if open-loop NLL improves only at H1, if
H2--H4 agreement does not exceed the registered threshold, or if direct
open-loop CE damages canonical NLL beyond the supporting bound.

## Evidence boundary

Passing these criteria does not establish calibrated stochastic diversity or
long-context sentence coherence. A later text-generation audit must remain a
separate record. Metrics are written to TSV; post-run interpretation is
written separately in Markdown.

## Stopped-run boundary

The user stopped this non-EMA run after the complete step-500 report to
replace the moving online closure teacher with a full-model EMA teacher. The
preserved final point is:

- validation teacher NLL: `4.934338`
- validation Open NLL: `6.194689`
- validation H2--H4 Open NLL: `6.612508`
- validation H2--H4 Open accuracy: `0.082804`
- validation H2--H4 online teacher/Open agreement: `0.244171`
- validation online closure KL: `1.480903`
- innovation RMS ratio: `0.498076`
- H4 open-state relative MSE: `6.376894`

This incomplete run remains a matched non-EMA comparison through step 500.
