# EXP-20260724 K=4 trajectory CE with h1 online MSE, microbatch 32

## Status

- State: queued before optimization update 1
- Authorization: user-requested follow-up run
- Final comparison point: step 1000
- Primary control:
  `../EXP-20260724-k4-ha-skew-stride1-batch64-micro32-three-trajectory-detached-ce-tau005-shared-13m/`

## Question

Does an attached relative-MSE constraint only on the first clean transition
`K hA -> hB_online` improve h1 alignment while retaining later-horizon noisy
trajectory competition?

## Objective

The matched three-trajectory objective remains
`Ci=sum_j CE(logits_ij,gold_j)`,
`wi=stopgrad(softmax_i(-Ci/0.05))`, and
`L_CE=mean_anchor sum_i wi Ci/4`.

This run adds only
`L_MSE=mean ||K hA-hB_online||^2/||hB_online||^2`,
with total loss `L=L_CE+L_MSE`. `hB_online` comes from the same current
encoder pass and is attached. No MSE is applied to h2 through h4.

## Fixed configuration

- WikiText-103 train/validation, BPE vocabulary 8192
- seed 1337; test split remains unread
- width 896, two reversible causal blocks
- exact inverse decoder, rms-tied head
- `K=exp(A-A^T)`, learned per-channel sigma initialized to 0.05
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
- versus the matched control, h1 confidence accuracy improves at least 0.005
- mean h2-h4 confidence accuracy falls by no more than 0.005
- responsibility maximum remains at least 0.80
- effective trajectory count remains at most 1.6
- mean sigma remains at least 0.01
- trajectory diversity remains nonzero at every horizon
