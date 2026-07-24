# EXP-20260724 K=4 three-trajectory detached CE, tau 0.05

## Status

- State: stopped by user after the complete step-1 report
- Authorization: user-requested replacement of the tau-1 run
- Final comparison point: step 1000
- Result status: implementation superseded by shared-computation run
- Primary comparison:
  `../EXP-20260724-k4-ha-skew-stride1-batch64-three-trajectory-detached-ce-13m/`

## Question

Was the failure of three noisy trajectories to compete caused by temperature
1.0 being much larger than the observed four-token CE-score gaps?

## Sole intended change

Trajectory responsibility temperature changes from `1.0` to `0.05`:

`Ci = sum_{j=1}^4 CE(logits_ij, gold_token_j)`

`wi = stopgrad(softmax_i(-Ci / 0.05))`

`L = mean_anchor sum_i wi Ci / 4`.

The reported `marginal_nll` field is therefore a temperature-scaled soft-min
CE value, not an ordinary normalized marginal likelihood. For comparison, the
evaluation also records `mc_marginal_nll_tau1`, which computes
`-log((1/3) sum_i exp(-Ci))/4` from the same three paths without changing the
training responsibility.

## Fixed configuration

- WikiText-103 train/validation, vocabulary 8192, seed 1337
- width 896, two reversible causal blocks
- exact shared inverse decoder and rms-tied head
- `K=exp(A-A^T)`, initialized identity and exactly orthogonal
- learned shared per-channel SigmaPredictor, initial sigma 0.05
- four horizons, three independent compounding-noise trajectories
- stride-one anchors 0 through 255
- effective batch 64 as four accumulated microbatches of 16
- hidden-state MSE disabled with weight 0
- AdamW, clip norm 1.0, strict float32, TF32 disabled
- 1000 updates using the existing 6000-update LR schedule horizon

## Split, checkpointing, and evaluation

- fixed validation starts: seed `1337+999`, 128 examples
- fixed validation noise stream: seed `1337+5151`
- reports at steps 1, 50, 100, 250, 500, 750, and 1000
- checkpoints preserved at 100, 250, 500, 750, and 1000
- checkpoint selection: minimum validation tau-0.05 soft-min CE
- test split remains unread

Gold-posterior/oracle metrics are diagnostic and unavailable at generation.
The target-free selector chooses the path with the highest sum of its own four
argmax-token log probabilities.

## Registered success criteria

At step 1000, useful nonuniform competition requires:

- mean maximum responsibility at least 0.80;
- mean effective trajectory count at most 1.6;
- each exchangeable trajectory index receives mean responsibility between
  0.28 and 0.39, ruling out permanent index collapse;
- mean sigma remains at least 0.01;
- token trajectory diversity is nonzero at all four horizons.

Deployable selection requires target-free confidence-selected mean h2 through
h4 accuracy to exceed mean-path accuracy by at least 0.005. Oracle-only gains
do not satisfy this criterion.

The tau-1 run stopped at step 500 and is a matched-step optimization trace, not
a completed 1000-step control. A positive outcome still requires a subsequent
matched one-trajectory CE-only control before becoming a wiki finding.

## Supersession note

This first tau-0.05 implementation duplicated the real-token encoder, clean K
orbit, sigma predictor, and inverse-decoder prefix computation across the
three trajectories. The user stopped it after step 1 and requested exact
shared computation. Its step-1 checkpoint is preserved with SHA-256
`13423c3a3b67163858d90b9aca3dfe09a456139afe347b9217300df5ddd0edd7`,
but it is implementation validation rather than experiment evidence.
