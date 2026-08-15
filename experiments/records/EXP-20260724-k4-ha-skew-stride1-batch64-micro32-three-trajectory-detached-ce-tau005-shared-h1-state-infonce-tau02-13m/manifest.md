# EXP-20260724 K=4 trajectory CE with attached h1 state-InfoNCE

## Status

- State: preregistered before optimization update 1
- Authorization: user-requested implementation and training
- Final comparison point: step 1000
- Primary controls:
  - CE-only:
    `../EXP-20260724-k4-ha-skew-stride1-batch64-micro32-three-trajectory-detached-ce-tau005-shared-13m/`
  - attached h1 MSE:
    `../EXP-20260724-k4-ha-skew-stride1-batch64-micro32-three-trajectory-detached-ce-tau005-shared-h1-online-mse-13m/`

## Question

Can direct attached symmetric state-InfoNCE align clean `K hA` with its
online `hB` while preventing the collective contraction allowed by h1 MSE,
preserving token accuracy and nontrivial trajectory noise?

## Sole objective change

The three-trajectory detached tau-0.05 CE objective is unchanged. Hidden-state
MSE is disabled. At h1, let `u_i=K hA_i` and `v_i=hB_i`, both attached.
For each anchor position, the other examples in the same physical microbatch
are negatives. Pair energy is the symmetric scale-invariant distance

`d_ij = ||u_i-v_j||^2 / sqrt(||u_i||^2 ||v_j||^2)`.

The added loss is the mean of `u->v` and `v->u` cross-entropy over
`-d/tau`, with diagonal matches positive, temperature 0.2, and weight 1.
There is no projection head, stop-gradient, EMA encoder, posterior model,
KL term, or second training stage.

## Fixed configuration

- WikiText-103 train/validation, BPE vocabulary 8192
- seed 1337; test split remains unread
- width 896, two reversible causal blocks
- exact inverse decoder, rms-tied head
- `K=exp(A-A^T)`, learned per-channel sigma initialized to 0.05
- horizons 4, trajectories 3, stride-one anchors 0 through 255
- trajectory temperature 0.05
- state-InfoNCE temperature 0.2, weight 1, h1 only
- effective batch 64, microbatch 32, accumulation steps 2
- state negatives are the other 31 examples within each microbatch
- strict float32, TF32 disabled
- AdamW, clip norm 1.0
- 1000 updates on the 6000-update LR schedule
- validation starts: 128 examples from seed `1337+999`
- validation noise stream: seed `1337+5151`

## Reporting and success criteria

- reports at 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at 100, 250, 500, 750, and 1000
- select minimum validation tau-0.05 soft-min CE
- h1 positive state distance is below negative distance
- training state retrieval exceeds random chance (`1/32`)
- h1 confidence accuracy does not fall below the CE-only control
- mean h2-h4 confidence accuracy falls by no more than 0.005
- responsibility maximum remains at least 0.80
- effective trajectory count remains at most 1.6
- mean sigma remains at least 0.01
- trajectory diversity remains nonzero at every horizon
