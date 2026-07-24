# EXP-20260724 K=4 trajectory CE with h1 online MSE

## Status

- State: planned; implementation only, no optimization update executed
- Authorization: user-requested
- Final comparison point: step 1000
- Primary control:
  `../EXP-20260724-k4-ha-skew-stride1-batch64-three-trajectory-detached-ce-tau005-shared-13m/`

## Question

Can one online, attached relative-MSE constraint on the shared clean
transition `K hA` improve h1 alignment while preserving the noisy
four-token trajectory competition at later horizons?

## Sole objective change

The three-trajectory tau-0.05 CE objective remains

`Ci = sum_j CE(logits_ij, gold_token_j)`,

`wi = stopgrad(softmax_i(-Ci / 0.05))`,

`L_CE = mean_anchor sum_i wi Ci / 4`.

The variant adds equal-weight relative MSE only at the first clean transition:

`L = L_CE + mean ||K hA - hB_online||^2 / ||hB_online||^2`.

`hB_online` comes from the same current encoder pass that creates `hA`.
It is not produced by an EMA encoder and is not detached. Therefore this
term can shape both `K hA` and the realized-token encoder geometry.

No MSE is applied to h2 through h4. The first-horizon clean state is shared
across all three trajectories and precedes local `epsilon_1`; later noisy
branches remain supervised only by trajectory CE.

## Fixed configuration

- WikiText-103 train/validation, vocabulary 8192, seed 1337
- width 896, two reversible causal blocks
- exact inverse decoder and rms-tied head
- `K=exp(A-A^T)`, initialized identity and exactly orthogonal
- learned per-channel SigmaPredictor, initial sigma 0.05
- horizons 4, trajectories 3, anchor stride 1
- effective batch 64, microbatch 16, four gradient accumulations
- strict float32, TF32 disabled
- AdamW, clip norm 1.0
- 1000 updates on the existing 6000-update LR schedule
- fixed validation starts from seed `1337+999`
- fixed validation noise stream from seed `1337+5151`
- test split remains unread

## Checkpointing and comparison

- reports at steps 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at 100, 250, 500, 750, and 1000
- selection by minimum validation tau-0.05 soft-min CE, matching the control
- primary comparisons are matched-step confidence/oracle h1 accuracy,
  h2-h4 accuracy, responsibility statistics, sigma, and trajectory diversity

## Success criteria

At step 1000, relative to the matched CE-only control:

- confidence-selected h1 accuracy improves by at least 0.005;
- mean confidence-selected h2-h4 accuracy falls by no more than 0.005;
- mean maximum responsibility remains at least 0.80 and effective trajectory
  count remains at most 1.6;
- mean sigma remains at least 0.01;
- trajectory diversity remains nonzero at every horizon.

Oracle-only improvement is diagnostic and does not establish deployable
selection. Any result is limited to this seed.
