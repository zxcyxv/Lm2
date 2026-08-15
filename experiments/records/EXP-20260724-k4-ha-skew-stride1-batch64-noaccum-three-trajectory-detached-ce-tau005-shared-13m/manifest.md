# EXP-20260724 K=4 trajectory competition, batch 64 without accumulation

## Status

- State: failed before optimization update 1 (CUDA OOM)
- Authorization: user-requested training run
- Final comparison point: step 1000
- Producer:
  `train_k4_ha_skew_three_trajectory_detached_ce_tau005_shared_batch64_noaccum_13m.py`
- Matched accumulated-batch reference:
  `../EXP-20260724-k4-ha-skew-stride1-batch64-three-trajectory-detached-ce-tau005-shared-13m/`

## Question

Does the tau-0.05 three-trajectory competition train successfully for 1000
updates when the full batch of 64 is processed in one forward/backward pass,
without gradient accumulation?

## Architecture and objective

Each example is encoded once by a width-896, two-block reversible causal
encoder. A learned exactly orthogonal `K=exp(A-A^T)` produces the shared clean
orbit `K hA` through `K^4 hA`; a per-channel SigmaPredictor supplies learned
noise scales. Three independent Gaussian-noise trajectories branch from that
shared computation. The exact inverse decoder shares prefix FFN, attention,
and prefix QKV/KV work, while noisy branch states and token readout retain the
trajectory dimension.

For trajectory `i`, `Ci` is the sum of the four future-token cross-entropies.
The optimization objective is

`wi = stopgrad(softmax_i(-Ci / 0.05))`,

`L = mean_anchor sum_i wi Ci / 4`.

No hidden-state MSE is optimized.

## Fixed configuration

- WikiText-103 train/validation, BPE vocabulary 8192
- seed 1337
- train split for optimization; validation split only for model selection
- test split remains unread
- width 896, two reversible causal blocks
- exact inverse decoder, rms-tied head
- horizons 4, trajectories 3, stride-one anchors 0 through 255
- physical batch 64, microbatch 64, one backward pass per update
- gradient accumulation disabled (`accumulation_steps=1`)
- strict float32, TF32 disabled
- AdamW, clip norm 1.0
- 1000 updates on the 6000-update learning-rate schedule
- fixed validation starts from seed `1337+999`, 128 examples
- fixed validation noise stream from seed `1337+5151`

## Reporting and success criteria

- reports at steps 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at 100, 250, 500, 750, and 1000
- select the minimum validation tau-0.05 soft-min CE
- at step 1000, mean maximum responsibility is at least 0.80
- effective trajectory count is at most 1.6
- each trajectory's mean responsibility is in `[0.28,0.39]`
- mean sigma is at least 0.01
- token diversity is nonzero at every horizon

## Result

Preflight passed with 13,770,624 trainable parameters and initial marginal CE
8.9764. The first training forward failed before update 1 with CUDA OOM on a
31.36 GiB device: 30.79 GiB was in use and the full-vocabulary CE requested
another 6.00 GiB. No checkpoint or metric row was produced. The user
subsequently authorized a microbatch-32, two-accumulation replacement run.
