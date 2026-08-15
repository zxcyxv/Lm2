# EXP-20260725 K=4 branch-operator trajectory commitment

## Status

- State: preregistered before optimization update 1
- Authorization: user-requested implementation and training
- Final comparison point: step 1000
- Primary control:
  `../EXP-20260725-k4-ha-skew-stride1-batch64-micro16-initial-condition-three-trajectory-prior-tau005-h1-online-mse-13m/`
- Test split remains unread

## Question

Does replacing the control's one-shot residual vector with one
context-conditioned orthogonal branch operator, selected once and reused at
every horizon, restore a common one-step law between AR and block rollout
without changing the training objective?

The control uses

`u_i1 = K hA + eta_i` and `u_ij = K^(j-1) u_i1`.

The proposal uses one fixed `T_i = Q_i K`:

`u_i0 = hA` and `u_i,j+1 = T_i u_ij`.

Thus `u_ij = T_i^j hA`; no branch operator is regenerated or reselected
inside the four-token trajectory.

## Architecture change

The reversible encoder, exact inverse decoder, rms-tied head, global
`K=exp(A-A^T)`, clean `K hA ... K^4 hA` orbit, prefix prior, branch codes,
seed, data, four-token sequence CE, and h1 online MSE remain unchanged.

For each prefix/branch, a shared bottleneck receives
`stopgrad(hA)` and fixed code `z_i`, and emits two vectors. Modified
Gram--Schmidt turns them into an orthonormal plane `(p_i,q_i)`. A
prefix-conditioned angle defines a plane rotation `Q_i`. The same plane and
angle are reused for every horizon:

`u_i,j+1 = Q_i K u_ij`.

Equivalently,

`u_i,j+1 = K u_ij + (Q_i-I) K u_ij`.

`Q_i` is applied in `O(width)` without materializing a dense matrix. Because
both `Q_i` and `K` are orthogonal, `T_i=Q_iK` is orthogonal. The initial angle
is chosen so an isotropic state has expected correction RMS ratio `0.05`,
matching the control's registered initial residual ratio.

Unlike the control's vector residuals, distinct rotations cannot have an
exact nontrivial prior-weighted arithmetic centroid equal to identity. No
replacement centering penalty or loss is introduced.

## Objective

The objective is byte-for-byte the same common persistent-trajectory loss
used by the initial-condition control:

`L_h1 = relMSE(K hA, hB_online)`,

`C_i = sum_j CE(logits_ij, gold_j)`,

`q_i = stopgrad softmax(log pi_i - C_i / 0.05)`,

`L_path = mean sum_i q_i C_i / 4`,

`L_prior = mean KL(q || pi)`,

`L = L_path + L_h1 + L_prior`.

No clean AR CE, branch-state MSE, all-horizon latent alignment, operator
energy penalty, canonicalizer, simplex map, innovation tape, or additional
loss is added.

## Fixed configuration

- WikiText-103 train/validation, BPE vocabulary 8192
- seed 1337; test split remains unread
- width 896, two reversible causal blocks
- exact inverse decoder, rms-tied head
- global `K=exp(A-A^T)`
- three branch operators and four horizons
- branch-code width 32; generator bottleneck width 64
- one context-conditioned two-dimensional rotation plane per branch
- expected initial branch correction RMS ratio 0.05
- one prefix-only prior over three branches
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

- every `Q_i` norm-preservation error below `1e-5`
- repeated-transition reconstruction
  `stack(T_i hA,T_i^2 hA,...)-forward_tape` below `1e-5`
- nonzero finite gradients to global `K`, branch plane, branch angle, prior,
  and attached h1 online target
- no gradient from h1 MSE to h2--h4 online targets
- distinct branches produce distinct states

## Step-1000 training criteria

- clean h1 relative MSE at most `0.03`
- validation marginal NLL at most `5.738737`, no more than `0.10` above the
  initial-condition control's `5.638737`
- prefix-prior winner accuracy above random chance `1/3`
- nonzero observed branch correction and token diversity at every horizon
- no non-finite training or validation metric

## Required post-training AR/block audit

The architecture claim requires a same-operator evaluation in which `Q_i` is
selected once and retained by both policies:

- block: use `T_i^1 ... T_i^4` before re-encoding
- AR: after each emitted token, replace only the semantic state with the
  actual-prefix encoder state and apply the same retained `T_i` again

The prior and `Q_i` must not be recomputed each AR token. On the existing 64
fixed validation prompts, registered evidence of improvement requires:

- same-operator AR/block token agreement at least `0.15`, versus the old
  mismatched prior-pair agreement `0.09155`;
- block period-4 repetition at most `0.4451`, at least `0.05` below the
  initial-condition `prior_block4` value `0.4951`; and
- collapsed-sample fraction at most `0.2719`, no more than `0.10` above the
  initial-condition value `0.1719`.

Failure of the generation thresholds with successful structural preflight
separates “the one-step law is now homogeneous” from “one fixed branch
operator is expressive enough to produce a coherent semantic trajectory.”

