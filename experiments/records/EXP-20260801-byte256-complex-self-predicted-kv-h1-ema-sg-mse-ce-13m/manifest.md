# EXP-20260801 EMA-SG H1 complex KV regression

## Status

- State: completed all 1000 optimizer updates
- Immediate control:
  `../EXP-20260801-byte256-complex-self-predicted-kv-h1-online-sg-mse-ce-13m/`
- Attached-target parent:
  `../EXP-20260801-byte256-complex-self-predicted-kv-full-innovation-h1-online-mse-ce-13m/`
- Test split remains unmaterialized and unread.

### Execution result

- The preregistered all-required criterion did not pass because mean state
  cosine was `0.495614 < 0.75` and the separate-process step-100 replay did
  not meet the `1e-5` tolerance.
- The agreement-specific threshold passed: EMA H2--H4 agreement was
  `0.272135`, above the attached-target parent's `0.261149`.
- Exact four-token block agreement was `0.024658`; the model is therefore not
  close to exact AR equivalence despite the average-agreement improvement.
- EMA H1 validation NLL was `1.393542`, and EMA H2--H4 no-write agreement was
  `0.171631`, below the write-enabled result.
- At step 100, every online/EMA validation metric was exactly equal within the
  new run because decay was zero. The failed replay criterion concerns the
  trajectory of the earlier separately launched SG run, not the EMA copy.

## Question

Does a warm-started full-model EMA stop-gradient target reduce coordinate
chasing enough to preserve the self-composed central path's agreement with the
same model's greedy AR path, while retaining H1 CE quality?

The inference model is the complete EMA snapshot. In particular its encoder
and analytic inverse decoder use the same EMA weights; the central transition
is also taken from that snapshot so no online/EMA latent coordinates are mixed.

## Fixed recurrence and readout

- byte vocabulary 256; width 1344; two reversible causal encoder blocks
- analytic exact-inverse decoder and fixed regular-simplex raw tied logits
- 8 complex-memory heads; complex key dimension 16; value dimension 31
- learned unit-modulus hidden and memory rotations; no decay
- coefficient-free self-predicted rank-one innovation
- no beta, temperature, gate, residual prior, independent vocabulary head,
  sampled branch, or observed/generated-token KV write

For one transition:

`h_prior = Read(q(h_rot), U S)`

`I = k(h_prior) v(h_prior)^dagger`

`h_post = h_prior + Read(q(h_prior), I)`

`S_next = U S + I`

Only `h_prior` is decoded for current-token CE. `h_post` is recurrent and is
the H1 latent prediction.

## EMA target and gradient routing

The online prediction is trained against a detached EMA encoder coordinate:

`L_H1 = ||h_post_online - sg(E_EMA(prefix,B))||^2`
`       / ||sg(E_EMA(prefix,B))||^2`.

Total loss:

`L = CE(raw_tied_logits(D_online(h_prior_online)), B) + L_H1`.

- CE and prediction-side MSE update the complete online model.
- The EMA target is evaluated under `no_grad`; no EMA parameter has a gradient
  or optimizer state.
- The future token appears only in the detached target branch and labels. It
  never enters the central recurrence.
- The EMA is a complete copy of encoder, transition, exact inverse decoder,
  tied head, and embedding.

For post-update steps 1--100, EMA decay is zero, so EMA is an exact online
copy. Beginning at step 101:

`theta_EMA <- 0.99 theta_EMA + 0.01 theta_online`.

All primary validation and AR-equivalence metrics use the complete EMA model:

`E_EMA -> T_EMA -> D_EMA = E_EMA^{-1}`.

Online-model metrics are retained as secondary diagnostics; their H1 latent
relative MSE uses the EMA encoder target, matching the training target, while
their AR-agreement metrics remain the online model's own central/AR comparison.

## Data and fixed execution

- WikiText-103 raw UTF-8 train and validation bytes only
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed `1337 + 999`
- context 256; training horizon 1; stride-one anchors
- monitoring horizon 4; boundary stride 4
- effective batch 64; physical microbatch 16
- strict float32; TF32 disabled
- AdamW, peak LR `3e-4`, clip norm 1.0
- 1000 updates on the fixed 6000-update schedule
- reports at 0, 1, 50, 100, 250, 500, 750, and 1000

## Structural preflight

- online and EMA state dicts are exactly equal initially
- EMA encoder targets equal the online-SG targets at initialization
- EMA target tensors and all EMA parameters are gradient-free
- CE and latent MSE retain finite nonzero gradients to the online encoder,
  complex projections, phases, and innovation readout
- held-out future-byte mutations do not affect the central tape or main logits
- H1 central and greedy-AR paths agree before feedback
- no beta parameter exists
- an EMA update with decay zero is an exact copy; decay `0.99` matches the
  registered affine update

## Success criteria

Step 100 must reproduce the stopped online-SG control within `1e-5` absolute
error for H1 NLL, H1 latent relative MSE, mean state cosine, and H2--H4 token
agreement.

At step 1000 all are required:

- EMA H1 validation NLL improves from step 0
- EMA H1 latent relative MSE improves from step 0
- EMA H1 posterior target accuracy exceeds `0.109619`
- EMA H2--H4 central/greedy-AR agreement exceeds `0.261149`
- EMA write-enabled H2--H4 agreement exceeds EMA no-write agreement
- EMA four-horizon mean central/greedy-AR state cosine is at least `0.75`
- EMA H1 central/AR max logit error remains below `5e-4`

Metrics are TSV and interpretation is Markdown. Online and EMA checkpoints
remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_h1_ema_sg_mse_ce_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2 \
  --record-dir \
    experiments/records/EXP-20260801-byte256-complex-self-predicted-kv-h1-ema-sg-mse-ce-13m \
  --output-dir \
    outputs/experiments/EXP-20260801-byte256-complex-self-predicted-kv-h1-ema-sg-mse-ce-13m
```

## Evidence boundary

This experiment tests deterministic greedy central/AR equivalence and latent
closure under one fixed seed and validation split. It does not establish
sampled joint-distribution equivalence or performance on the unread test split.
