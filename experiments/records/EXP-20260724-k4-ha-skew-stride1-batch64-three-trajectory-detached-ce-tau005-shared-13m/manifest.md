# EXP-20260724 K=4 three-trajectory tau-0.05, shared computation

## Status

- State: preregistered before main-run optimization update 1
- Authorization: user-requested corrected rerun
- Final comparison point: step 1000
- Supersedes:
  `../EXP-20260724-k4-ha-skew-stride1-batch64-three-trajectory-detached-ce-tau005-13m/`

## Question

Does tau-0.05 trajectory competition persist through 1000 updates when all
deterministic computation is shared exactly across the three noisy paths?

## Shared computation

Each original example is encoded once. The following are also computed once
and shared across trajectory samples:

- real-token encoder states;
- clean `K^1 hA` through `K^4 hA` orbit;
- SigmaPredictor output;
- inverse-decoder prefix FFN, prefix attention, and prefix QKV/KV.

Only Gaussian noise, propagated noisy branch states, branch QKV/attention, and
token readout carry a trajectory dimension. A literal expanded-prefix
reference produced matching logits and gradients within floating-point
tolerance; see [`implementation-smoke.md`](implementation-smoke.md).

## Objective

For three paths and four gold future tokens,

`Ci = sum_{j=1}^4 CE(logits_ij, gold_token_j)`,

`wi = stopgrad(softmax_i(-Ci / 0.05))`,

`L = mean_anchor sum_i wi Ci / 4`.

Hidden-state MSE is absent (`mse_weight=0`). The `marginal_nll` metric is a
tau-0.05 soft-min CE, not ordinary likelihood. `mc_marginal_nll_tau1` records
the temperature-one Monte-Carlo marginal diagnostic from the same paths.

## Fixed configuration

- WikiText-103 train/validation, vocabulary 8192, seed 1337
- width 896, two reversible causal blocks
- exact inverse decoder, rms-tied head
- `K=exp(A-A^T)`, initialized identity and exactly orthogonal
- learned per-channel SigmaPredictor, initial sigma 0.05
- horizons 4, trajectories 3, stride-one anchors 0 through 255
- effective batch 64, microbatch 16, four gradient accumulations
- strict float32, TF32 disabled
- AdamW, clip norm 1.0
- 1000 updates on the existing 6000-update LR schedule horizon

Microbatch 32 is not used: even after eliminating redundant shared-state
copies, exact 8192-way CE exceeded the 16 GB device during the registered
implementation smoke.

## Split and evaluation

- fixed validation starts: seed `1337+999`, 128 examples
- fixed validation noise stream: seed `1337+5151`
- reports at 1, 50, 100, 250, 500, 750, and 1000
- checkpoints preserved at 100, 250, 500, 750, and 1000
- selection: minimum validation tau-0.05 soft-min CE
- test split remains unread

## Success criteria

At step 1000:

- mean maximum responsibility at least 0.80;
- mean effective trajectory count at most 1.6;
- each exchangeable trajectory index has mean responsibility in `[0.28,0.39]`;
- mean sigma at least 0.01;
- token diversity is nonzero at all horizons.

Deployable selection additionally requires target-free confidence-selected
mean h2 through h4 accuracy to exceed mean-path accuracy by at least 0.005.
Gold-oracle gains alone are insufficient. Any positive result remains limited
to this seed and requires a later matched one-trajectory CE-only control.
