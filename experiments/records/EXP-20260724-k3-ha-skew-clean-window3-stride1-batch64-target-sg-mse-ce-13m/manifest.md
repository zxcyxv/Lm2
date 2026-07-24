# EXP-20260724 K=3 stride-1 gold-target stop-gradient

## Status

- State: completed at step 100
- Authorization: user-requested
- Primary comparison:
  `../EXP-20260724-k3-ha-skew-clean-window3-stride1-batch64-mse-ce-13m/`

## Question and sole gradient intervention

Does applying stop-gradient only to the gold `hB/hC/hD` targets of the
relative-MSE term improve the first 100 updates?

The CE graph and the predicted orbit remain fully differentiable:
`z1=K(hA)`, `z2=K(z1)`, and `z3=K(z2)`. Later CE and MSE terms retain all
backward paths through earlier orbit states and the shared K. Only

`MSE(zj, hj)` becomes `MSE(zj, sg(hj))`.

All other configuration is retained:

- WikiText-103 train/validation, vocabulary 8192, seed 1337
- batch 64, context 256, anchor stride 1, horizons 3
- width 896 and two reversible causal blocks
- exact shared inverse decoder, rms-tied head, `K=exp(A-A^T)`
- no noise, sigma predictor, spectral constraint, or horizon truncation
- mean h1/h2/h3 CE plus mean relative MSE
- AdamW, 100-step warmup schedule, gradient clip 1.0

The run stops at step 100. These first 100 learning rates exactly match the
first 100 updates of the 6000-step attached-target control because all are
inside the shared 100-step linear warmup.

## Split, selection, and evaluation

- fixed validation starts: seed `1337+999`, 128 examples
- fixed h1..h5 rollout starts: seed `1337+7777`, 256 examples
- reports at steps 1, 50, and 100
- checkpoint selection: minimum mean validation h1..h3 NLL
- test split remains unread

## Registered success criteria

At step 100, relative to the attached-target control:

- validation h1 accuracy improves by at least 0.005 over `0.092529`;
- mean validation h2/h3 accuracy falls by no more than 0.005 from `0.054291`;
- mean validation h1..h3 NLL is below `7.122284`.

Any conclusion is limited to early optimization at this seed.

## Result

The preserved step-100 `last.pt` has SHA-256
`07a2052a9c73f26c043239cf50dec229ef028591fe391b42489956c3ccaa4b13`.

Relative to the attached-target control at step 100:

- validation h1/h2/h3 accuracy changed from
  `0.092529/0.057556/0.051025` to
  `0.065704/0.045959/0.039764`;
- mean validation NLL changed from `7.122284` to `7.163808`;
- h1/h2/h3 relative MSE changed from
  `0.012155/0.022225/0.037473` to
  `0.049672/0.128638/0.238277`.

All registered success criteria failed. In this early matched run, target-only
stop-gradient is refuted as an optimization improvement.
