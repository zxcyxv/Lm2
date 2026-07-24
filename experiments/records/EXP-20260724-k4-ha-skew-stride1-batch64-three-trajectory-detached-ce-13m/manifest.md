# EXP-20260724 K=4 three-trajectory detached CE competition

## Status

- State: stopped by user after the complete step-500 report
- Authorization: user-requested
- Final comparison point: step 1000
- Intermediate reports at 100/250/500/750 are trend evidence only
- Result status: incomplete; superseded by the tau-0.05 run

## Question

Can three independently noised four-token latent trajectories avoid the
conditional-mean behavior of a single trajectory when the trajectory whose
four gold tokens have the lowest summed CE receives the largest detached
training responsibility?

This is a new latent-mixture training mechanism. It is not eligible for a
100-step conclusion.

## Registered objective

For each real anchor state `hA`, construct three independent compounding-noise
trajectories with four decoder slots. Noise scale at horizon `j` is predicted
from the clean `K^j hA`; Gaussian draws are independent across trajectory,
horizon, anchor, and example.

For trajectory `i`,

`Ci = sum_{j=1}^4 CE(logits_ij, gold_token_j)`.

Responsibilities and the CE-only loss are

`wi = stopgrad(softmax_i(-Ci / 1.0))`

and

`L = mean_anchor sum_i wi Ci / 4`.

The logged scalar value is the equivalent temperature-one Monte-Carlo
marginal NLL,

`-mean_anchor log((1/3) sum_i exp(-Ci)) / 4`,

while its gradient is implemented explicitly through detached
responsibilities. The responsibility selector is not a gradient path; each
trajectory CE remains differentiable through the inverse decoder, noise
predictor, `K`, and encoder.

Hidden-state MSE is disabled with registered weight `0`. The common objective
retains an optional MSE parameter for a later ablation, but no MSE term enters
this run's optimization.

## Fixed configuration

- WikiText-103 train/validation, vocabulary 8192, seed 1337
- width 896, two reversible causal blocks
- exact shared inverse decoder and rms-tied head
- `K=exp(A-A^T)`, initialized to identity and exactly orthogonal
- learned shared per-channel SigmaPredictor, initial sigma 0.05
- four horizons, three trajectories, stride-one anchors 0 through 255
- effective batch 64 as four accumulated microbatches of 16
- AdamW, clip norm 1.0, strict float32 with TF32 disabled
- 1000 optimizer updates using the existing 6000-update LR schedule horizon,
  so step 1000 remains continuable and update-LR comparable to the earlier
  long runs

## Split and evaluation

- fixed validation starts: seed `1337+999`, 128 examples
- fixed validation noise stream: seed `1337+5151`
- reports at steps 1, 50, 100, 250, 500, 750, and 1000
- checkpoint selection: minimum validation three-sample marginal NLL
- preserved checkpoints at 100, 250, 500, 750, and 1000
- test split remains unread

Validation reports:

- three-sample marginal and mean-path NLL;
- gold-posterior responsibility, entropy, effective trajectory count, and
  trajectory-index usage;
- path-average, gold-oracle-selected, and target-free
  confidence-selected h1 through h4 accuracy;
- token diversity, learned sigma, and hidden-state MSE as diagnostics only.

The gold-oracle selector is not a deployable generation result. The
target-free selector chooses the trajectory with the highest sum of its own
four argmax-token log probabilities.

## Comparison and success criteria

The internal compute-matched reference is the mean of the same model's three
individual path NLLs. Historical clean and learned-noise CE+MSE window-3 runs
are descriptive only because both the horizon and objective differ:

- `../EXP-20260724-k3-ha-skew-clean-window3-stride1-batch64-mse-ce-13m/`
- `../EXP-20260724-k3-ha-skew-window3-stride1-batch64-learned-compounding-noise-mse-ce-13m/`

At step 1000, evidence for useful trajectory competition requires all of:

- marginal NLL at least 0.02 below mean-path NLL;
- mean maximum responsibility at least 0.50 while each trajectory-index mean
  responsibility remains between 0.28 and 0.39;
- mean effective trajectory count below 2.7, showing sample-dependent
  specialization rather than uniform weighting;
- mean learned sigma at least 0.01 and nonzero token diversity at every
  horizon.

Deployable selection is separately supported only if target-free
confidence-selected mean h2 through h4 accuracy exceeds mean-path accuracy by
at least 0.005. Oracle improvement alone is not sufficient.

A positive result remains limited to this seed and must subsequently be
checked against a separately trained matched one-trajectory CE-only control.

## Stopped-run result

At step 500, temperature 1.0 had effectively disabled competition:
mean maximum responsibility was `0.347416`, effective trajectory count was
`2.990919`, and validation temperature-one marginal/mean-path NLL was
`6.216308/6.217945`. Mean sigma was `0.032693`. The last complete checkpoint
is `step0500.pt`, SHA-256
`71c9688681836fb9db696a9573a18a1c06bd1acb2f566e45916fc8a6a12f468d`.

This stopped result does not answer the registered step-1000 question. It is
preserved as evidence that the observed four-token CE gaps were too small
relative to temperature 1.0 to produce non-uniform responsibilities.
