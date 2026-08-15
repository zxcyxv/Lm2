# EXP-20260724 K=4 trajectory competition, batch 64 microbatch 32

## Status

- State: preregistered before optimization update 1
- Authorization: user-requested replacement after physical-batch OOM
- Final comparison point: step 1000
- Failed predecessor:
  `../EXP-20260724-k4-ha-skew-stride1-batch64-noaccum-three-trajectory-detached-ce-tau005-shared-13m/`

## Question

Does tau-0.05 three-trajectory competition train successfully for 1000
updates at effective batch 64 using microbatch 32 and two accumulations?

## Architecture and objective

A width-896, two-block reversible causal encoder processes each prefix. The
shared clean orbit `K hA` through `K^4 hA` uses the learned orthogonal
`K=exp(A-A^T)` and per-channel SigmaPredictor. Three independent noisy
trajectories retain separate branch states and token readouts while sharing
the encoder, clean orbit, sigma tape, and inverse-decoder prefix computation.

For each path, `Ci` is its four-token CE sum:
`wi=stopgrad(softmax_i(-Ci/0.05))` and
`L=mean_anchor sum_i wi Ci/4`. Hidden-state MSE is disabled.

## Fixed configuration

- WikiText-103 train/validation, BPE vocabulary 8192
- seed 1337; test split remains unread
- horizons 4, trajectories 3, stride-one anchors 0 through 255
- effective batch 64, microbatch 32, accumulation steps 2
- strict float32, TF32 disabled
- AdamW, clip norm 1.0
- 1000 updates on the 6000-update LR schedule
- validation starts: 128 examples from seed `1337+999`
- validation noise stream: seed `1337+5151`

## Reporting and success criteria

- reports at 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at 100, 250, 500, 750, and 1000
- select minimum validation tau-0.05 soft-min CE
- step-1000 responsibility maximum at least 0.80
- effective trajectory count at most 1.6
- each trajectory responsibility in `[0.28,0.39]`
- mean sigma at least 0.01
- nonzero token diversity at every horizon
