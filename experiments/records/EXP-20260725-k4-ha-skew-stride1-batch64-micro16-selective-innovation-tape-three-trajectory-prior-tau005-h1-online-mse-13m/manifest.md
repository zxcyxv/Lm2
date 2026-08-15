# EXP-20260725 K=4 selective innovation-tape trajectories

## Status

- State: preregistered before optimization update 1
- Authorization: user-requested implementation and training
- Primary comparison point: step 1000
- Primary control:
  `../EXP-20260725-k4-ha-skew-stride1-batch64-micro16-initial-condition-three-trajectory-prior-tau005-h1-online-mse-13m/`
- Secondary diagnostic:
  `../EXP-20260725-k4-ha-skew-stride1-batch64-micro16-operator-commitment-three-trajectory-prior-tau005-h1-online-mse-13m/`
- Test split remains unread

## Question

Does replacing a single transported initial residual with a
noise-conditioned, causally scanned innovation tape reduce phase-coded
trajectory failure while preserving the global semantic operator and the
existing training objective?

The control uses

`u_i1 = K hA + eta_i` and `u_ij = K^(j-1) u_i1`.

The proposal samples one complete base-noise tape per candidate trajectory,
then deterministically compiles it into local innovations:

`xi_i,1:H ~ Normal(0,I)`,

`r_i,1:H = R_scan(stopgrad(hA), xi_i,1:H)`,

`u_i,0 = hA`,

`u_i,j+1 = K u_i,j + r_i,j+1`.

Conditional on the sampled tape, the entire trajectory is deterministic.
The same trajectory index is scored by the sum of all four token losses.

## Innovation compiler

For each prefix, trajectory, and horizon, a shared context/noise feature
produces diagonal affine scan coefficients:

`a_i,j = sigmoid(W_a f_i,j + b_a)`,

`b_i,j = (1-a_i,j) tanh(W_b f_i,j)`,

`m_i,j = a_i,j * m_i,j-1 + b_i,j`, with `m_i,0 = 0`.

The recurrence is evaluated by an associative diagonal-affine prefix scan.
One shared output map gives raw innovations:

`raw_r_i,j = W_r m_i,j`.

There are no horizon-specific heads. A candidate score derived from the
final scan state gives a prior over the sampled trajectory candidates.
Innovations are scaled to the control's registered initial RMS ratio and
centered at every horizon:

`sum_i stopgrad(pi_i) r_i,j = 0`.

Thus the clean `K` orbit remains the candidate-mixture center. The innovation
is an additive latent transition residual, not a rotation of `K h`.

## Objective

The loss is unchanged from the persistent initial-condition control:

`L_h1 = relMSE(K hA, hB_online)`,

`C_i = sum_j CE(logits_ij, gold_j)`,

`q_i = stopgrad softmax(log pi_i - C_i / 0.05)`,

`L_path = mean sum_i q_i C_i / 4`,

`L_prior = mean KL(q || pi)`,

`L = L_path + L_h1 + L_prior`.

No clean AR CE, branch-state MSE, state contrastive loss, operator penalty,
canonicalizer, simplex map, branch rotation, or horizon-specific objective is
added.

## Fixed configuration

- WikiText-103 train/validation, BPE vocabulary 8192
- seed 1337; test split remains unread
- width 896, two reversible causal blocks
- exact inverse decoder, rms-tied head
- global `K=exp(A-A^T)`
- three sampled trajectories and four horizons
- base-noise width 32 per trajectory/horizon
- selective scan state width 64
- expected initial innovation RMS ratio 0.05
- minimum/maximum learned scale 0.001/0.25
- one sampled-candidate prior
- trajectory posterior temperature 0.05
- clean h1 attached online relative-MSE weight 1
- prior KL weight 1
- stride-one anchors 0 through 255
- effective batch 64, physical microbatch 16, four accumulations
- strict float32, TF32 disabled
- AdamW, clip norm 1.0
- 1000 updates on the 6000-update LR schedule
- validation starts: 128 examples from seed `1337+999`
- reports at 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at 100, 250, 500, 750, and 1000

## Structural preflight criteria

- associative selective scan matches the sequential affine recurrence below
  `1e-6`
- identical random-generator state produces identical innovation tapes
- different noise tapes produce distinct trajectories
- prior-weighted innovation centroid below `1e-5` at every horizon
- explicit `K u + r` reconstruction matches the forward tape below `1e-5`
- finite nonzero gradients reach global `K`, selective scan, innovation
  output, sampled-candidate prior, and attached online h1 target
- no h1-MSE gradient reaches online h2--h4 targets
- global `K` is identity/orthogonal within `1e-4` at initialization

## Step-1000 training criteria

- clean h1 relative MSE at most `0.03`
- validation marginal NLL at most `5.738737`, no more than `0.10` above the
  initial-condition control's `5.638737`
- sampled-candidate prior winner accuracy above random chance `1/3`
- finite nonzero innovation scale and token diversity at every horizon
- no non-finite training or validation metric

## Evidence boundary

Passing the optimization criteria does not establish sentence coherence or
long-horizon parallel generation. A post-training generation audit must reuse
the identical sampled noise tape in the re-anchored and open-loop policies.
Their only allowed difference is replacement of the semantic state by the
actual generated-prefix encoder state.
