# EXP-20260726 selective innovation: joint NLL + behavioral closure

## Status

- State: preregistered before optimization update 1
- Authorization: user approved after loss-design discussion
- Primary comparison point: step 1000
- Primary control:
  `../EXP-20260725-k4-ha-skew-stride1-batch64-micro16-selective-innovation-tape-three-trajectory-prior-tau005-h1-online-mse-13m/`
- Long-run reference:
  `../EXP-20260725-k4-ha-skew-stride1-batch64-micro16-selective-innovation-tape-three-trajectory-prior-tau005-h1-online-mse-6000step-13m/`
- Test split remains unread

## Question

Does moving supervision from latent-point matching and detached winner
classification to

1. a proper probability-one joint trajectory mixture likelihood, and
2. same-tape predictive closure after the observed first future token,

improve calibrated selective trajectories without changing the architecture?

The experiment tests the loss only. The global orthogonal operator, exact
inverse decoder, selective innovation compiler, number of candidates, horizon,
data, seed, and optimizer schedule are fixed to the selective-innovation
control.

## Fixed trajectory architecture

For candidate `i`:

`xi_i,1:H ~ Normal(0,I)`,

`r_i,1:H = R_scan(stopgrad(hA), xi_i,1:H)`,

`u_i,0 = hA`,

`u_i,j = K u_i,j-1 + r_i,j`.

The whole innovation tape is sampled once. The same candidate index is kept
for all horizons. Innovations remain prior-centered at every horizon, and
`K=exp(A-A.T)` remains one global orthogonal operator.

## Proper joint trajectory likelihood

Let

`C_i = sum_j CE(logits_open_i,j, gold_j)`

be the negative log probability of the complete observed four-token path
under candidate `i`. The learned target-free candidate prior is `pi_i`.

The primary loss is the normalized finite-mixture negative log likelihood:

`L_joint = mean[-logsumexp_i(log pi_i - C_i)] / H`.

This is probability temperature 1. There is no detached responsibility in the
loss and no separate posterior-to-prior KL. The exact posterior

`q_i = softmax_i(log pi_i - C_i)`

is detached only when it is reported or used to weight the auxiliary closure
term.

The mixture is outside the complete path product. Candidate identity cannot
change between horizons.

## Same-tape behavioral closure

The observed first future state is

`hB = E(x_<=A, gold_1)`.

For every candidate, a reanchored comparison path replaces only the first
open-loop state:

`v_i,1 = hB`,

`v_i,j = K v_i,j-1 + stopgrad(r_i,j)`, for `j=2..H`.

The reanchored teacher and open-loop student use the identical candidate and
the identical already-sampled remaining innovation tape. No innovation,
candidate prior, or branch is resampled. Teacher construction and teacher
logits are stop-gradient.

Closure starts at horizon 2; `Head(D(hB))` is deliberately not used as a
teacher for the already-observed token B. For horizons 2 through H:

`L_close_i = mean_j KL(stopgrad(p_reanchor_i,j) || p_open_i,j)`.

The candidate-weighted term is

`L_close = mean sum_i stopgrad(q_i) L_close_i`.

The registered total objective is

`L = L_joint + 1.0 * L_close`.

## Explicit exclusions

- no latent MSE in the objective
- no state or trajectory InfoNCE
- no Gaussian latent NLL
- no `Head(D(hB))` token-prototype KL at horizon 1
- no clean-orbit token CE
- no detached low-temperature best-of-three path loss
- no separate candidate-prior KL
- no entropy or diversity regularizer
- no operator, decoder, innovation, or generation-policy change

Latent relative MSE remains a diagnostic metric only.

## Fixed configuration

- WikiText-103 train/validation, BPE vocabulary 8192
- seed 1337; test split remains unread
- width 896, two reversible causal blocks
- exact inverse decoder, RMS-tied head
- global `K=exp(A-A.T)`
- three sampled trajectories and four horizons
- base-noise width 32 per trajectory/horizon
- selective scan state width 64
- expected initial innovation RMS ratio 0.05
- minimum/maximum learned scale 0.001/0.25
- trajectory likelihood temperature 1
- behavioral-closure weight 1
- behavioral-closure horizons 2, 3, and 4
- stride-one anchors 0 through 255
- effective batch 64, physical microbatch 16, four accumulations
- strict float32, TF32 disabled
- AdamW, clip norm 1.0
- 1000 updates on the 6000-update LR schedule
- validation starts: 128 examples from seed `1337+999`
- reports at 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at 100, 250, 500, 750, and 1000

## Structural preflight criteria

- computed joint NLL matches an explicit
  `-log(sum_i pi_i * exp(-C_i)) / H` reference
- exact posterior responsibilities sum to one and are detached only in the
  reported/closure-weight view
- finite nonzero joint-NLL gradients reach global `K`, innovation compiler,
  innovation output, scale, and candidate prior
- closure teacher logits and gold `hB` receive no gradient
- finite nonzero closure gradients reach the open-loop trajectory
- reanchored recurrence uses `hB` once and exactly the open path's `r_i,2:H`
- closure excludes horizon 1 and is finite and nonnegative
- associative selective scan and innovation-centering invariants from the
  control continue to pass

## Step-1000 criteria

Primary:

- validation learned-prior, temperature-1 joint mixture NLL is below the
  control's step-1000 `5.917968`

Supporting:

- learned-prior joint NLL is no worse than uniform-prior joint NLL
- validation behavioral-closure KL is lower than its step-1 value
- finite nonzero innovation scale and token diversity remain at every horizon
- posterior effective trajectory count remains above `1.10`
- no non-finite training or validation metric

Prior winner accuracy against the single observed continuation is diagnostic,
not a success criterion.

## Post-training evidence boundary

Passing likelihood and closure criteria does not establish sentence
coherence. A generation audit must reuse the identical sampled noise tape and
candidate across reanchored and open-loop policies. It must report immediate
repeat, period-4 repeat, repeated 4-gram, collapsed-candidate rate, and
distinct-token metrics separately from validation TSV metrics.
