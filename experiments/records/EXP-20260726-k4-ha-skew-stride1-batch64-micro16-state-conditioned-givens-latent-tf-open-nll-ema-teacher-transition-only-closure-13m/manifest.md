# EXP-20260726 state-conditioned open-NLL with EMA closure teacher

## Status

- State: stopped after the complete step-100 report
- Authorization: user requested an EMA teacher after observing the online
  teacher/open closure target continue to separate
- Primary matched control:
  `../EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-open-nll-transition-only-closure-13m/`
- Original control:
  `../EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-nll-closure-13m/`
- Test split remains unread

## Question

Does replacing the per-update online canonical closure target with a slowly
moving full-model EMA target prevent the teacher distribution from outrunning
the self-fed latent rollout?

The primary matched control added direct H2--H4 open-loop token NLL and routed
behavioral KL only into the central transition. At step 250 it improved
open-loop token behavior relative to the original control, but closure KL
still rose after its early minimum. This experiment changes only the closure
teacher dynamics.

## Fixed architecture

The online architecture and central recurrence are identical to the matched
control:

`T_online(h) = K_online(h)h + R_online(h)`.

- state-conditioned two-stage exact Givens operator
- deterministic state-conditioned innovation
- canonical latent teacher forcing for direct one-step token NLL
- one sequential self-fed latent rollout for inference-equivalent Open NLL
- no token or embedding feedback in the central recurrence
- no sampled noise, candidates, prior, plan, latent MSE, or InfoNCE

## EMA closure teacher

One complete EMA copy contains the encoder, state-conditioned transition,
exact-inverse decoder, RMS-tied vocabulary head, and embedding.

Before optimizer update `s`, the EMA copy produces canonical teacher
probabilities from the observed gold prefixes under `no_grad`:

`p_EMA,j = softmax(logits_TF,j(theta_EMA))`.

The online model produces the self-fed latent trajectory and student
probabilities:

`p_OL,j = softmax(logits_OL,j(theta_online))`.

After the online optimizer step:

`theta_EMA <- 0.99 * theta_EMA + 0.01 * theta_online`.

The EMA is initialized as an exact copy of the online model before update 1.
Non-floating buffers are copied exactly. The EMA has no optimizer and no
trainable gradient path.

## Objective and routing

Direct online-model token objectives are unchanged:

`L_TF = mean_{B:E} CE(logits_TF,j, gold_j)`,

`L_OL = mean_{C:E} CE(logits_OL,j, gold_j)`.

EMA behavioral closure is:

`L_close_EMA = mean_{C:E} KL(p_EMA,j || p_OL,j)`.

At optimizer update `s`:

`L = L_TF + L_OL + lambda_s * L_close_EMA`,

`lambda_s = min(1.0, s / 100)`.

Gradient routing is fixed:

- `L_TF + L_OL` updates the complete online model.
- `L_close_EMA` accumulates gradients only in the online central transition.
- no closure gradient reaches the online encoder, inverse decoder, head, or
  embedding;
- no gradient reaches any EMA parameter.

The online canonical teacher distribution remains a diagnostic. Both
`KL(p_online_TF || p_OL)` and `KL(p_EMA || p_OL)` are reported so a lower EMA
loss cannot hide increasing disagreement with the current canonical model.

## Fixed training configuration

- WikiText-103 train and validation; BPE vocabulary 8192
- seed 1337; validation-start seed `1337 + 999`
- the same 128 validation starts as both controls
- width 896, two reversible causal blocks
- exact inverse decoder and RMS-tied vocabulary head
- transition bottleneck 128; two complete Givens stages
- four horizons; stride-one anchors 0 through 255
- effective batch 64, physical microbatch 16, four accumulations
- strict float32; TF32 disabled
- AdamW, clip norm 1.0
- 1000 optimizer updates on the same 6000-update LR schedule
- EMA decay `0.99`, updated after every online optimizer step
- closure-weight warmup over updates 1--100
- reports at 1, 50, 100, 250, 500, 750, and 1000
- checkpoints contain both online and EMA states at 100, 250, 500, 750, and
  1000
- test split remains unread

## Structural preflight criteria

- online and EMA parameters are exactly equal before update 1
- one synthetic EMA update matches the registered `0.99/0.01` equation
- direct TF/Open token NLL has no EMA gradient path
- EMA closure has finite nonzero gradients in the online angle and innovation
  heads
- routed EMA closure accumulates no gradient in any online non-transition
  parameter
- every EMA parameter remains gradient-free
- the EMA teacher probabilities are detached
- H1 remains excluded from Open NLL and both closure diagnostics
- future gold changes do not affect the online self-fed states or logits
- conditional Givens norm error remains below `1e-5`

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_k4_ha_skew_state_conditioned_givens_latent_tf_nll_closure_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --experiment-id \
    EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-open-nll-ema-teacher-transition-only-closure-13m \
  --open-nll-weight 1.0 \
  --closure-weight 1.0 \
  --closure-warmup-steps 100 \
  --closure-transition-only \
  --closure-teacher-ema-decay 0.99 \
  --record-dir \
    experiments/records/EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-open-nll-ema-teacher-transition-only-closure-13m \
  --output-dir \
    outputs/experiments/EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-open-nll-ema-teacher-transition-only-closure-13m
```

## Step-1000 success and failure criteria

Against the matched non-EMA control:

- EMA closure KL is lower at step 1000
- current-online TF/Open KL is lower at step 1000
- online H2--H4 top-1 agreement is higher at step 1000
- H2--H4 Open NLL is no more than `0.05` worse
- overall Open NLL is no more than `0.05` worse

Against the original control:

- overall Open NLL is below `5.959662`
- H2--H4 Open NLL is below `6.456272`
- H2--H4 online teacher/open agreement is at least `0.286715`

Supporting:

- teacher-forced NLL is at most `4.611288`
- all losses, EMA values, learned angles, and innovations remain finite
- open-loop accuracy is nonzero at every horizon
- conditional-operator norm error remains below `1e-5`

The central question fails if only EMA-target agreement improves while
current-online teacher/open agreement or Open NLL degrades beyond the
registered tolerances.

## Evidence boundary

EMA stabilizes a behavioral target; it does not add stochastic branch
sampling or prove exact autoregressive joint-distribution equivalence. Text
generation quality remains a separate audit. Metrics are stored in TSV and
post-run interpretation in a separate Markdown file.

## Stopped-run result

The cold-start EMA retained `0.99^100 = 0.366032` of the random
initialization at step 100. The EMA target had not become a competent
canonical teacher:

- validation teacher NLL: `6.904690`
- validation Open NLL: `7.145589`
- validation H2--H4 Open NLL: `7.225443`
- validation H2--H4 Open accuracy: `0.044830`
- EMA closure KL: `1.248965`
- current-online closure KL: `0.250272`
- EMA teacher/Open H2--H4 top-1 agreement: `0.007884`
- current-online H2--H4 top-1 agreement: `0.399160`

The user-authorized EMA direction is retained, but this cold-start schedule
was stopped because it actively supplied a stale, mostly initial teacher.
It is superseded by a separately registered warm-start EMA run.
