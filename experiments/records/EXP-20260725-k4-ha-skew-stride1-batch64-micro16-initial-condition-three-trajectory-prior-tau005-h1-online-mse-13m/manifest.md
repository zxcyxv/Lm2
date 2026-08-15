# EXP-20260725 K=4 initial-condition trajectory commitment

## Status

- State: preregistered before optimization update 1
- Authorization: user-requested implementation and training
- Final comparison point: step 1000
- Primary controls:
  - CE-only compounding-noise trajectory:
    `../EXP-20260724-k4-ha-skew-stride1-batch64-micro32-three-trajectory-detached-ce-tau005-shared-13m/`
  - attached h1 online-MSE compounding-noise trajectory:
    `../EXP-20260724-k4-ha-skew-stride1-batch64-micro32-three-trajectory-detached-ce-tau005-shared-h1-online-mse-13m/`

## Question

Does replacing independent per-horizon Gaussian innovations with one learned,
context-conditioned initial-condition code per trajectory preserve branch
identity under the shared orthogonal `K`, while a prefix-only prior learns to
select the gold-compatible four-token trajectory without harming clean h1
state alignment?

## Architecture change

The reversible encoder, exact inverse decoder, rms-tied head, global
`K=exp(A-A^T)`, clean `K hA ... K^4 hA` orbit, seed, data, and four-token
sequence CE remain unchanged.

For three fixed learned branch codes `z_i`, a shared bottleneck initializer
produces one context-conditioned residual per anchor:

`eta_i = R(stopgrad(hA), z_i)`.

A prefix-only prior head produces `pi_i=softmax(P(stopgrad(hA)))`. Residuals
are centered so `sum_i pi_i eta_i=0`. Each trajectory is initialized once:

`u_i1 = K hA + eta_i`,

and receives no later innovation:

`u_ij = K^(j-1) u_i1`.

Thus orthogonality preserves pairwise branch distances across every horizon.
The initializer has no horizon input, token-prefix attention, or
horizon-specific parameters.

## Objective

The clean center receives the existing attached online h1 relative MSE:

`L_h1 = relMSE(K hA, hB_online)`.

For each branch, `C_i` is the sum of its four token cross-entropies. Detached
posterior responsibility uses the preregistered temperature `tau=0.05`:

`q_i = stopgrad softmax(log pi_i - C_i / tau)`.

The trajectory and prior losses are

`L_path = mean sum_i q_i C_i / 4`,

`L_prior = mean KL(q || pi)`,

and the total loss is

`L = L_path + L_h1 + L_prior`.

No branch-state MSE, `P` canonicalizer, simplex map, dynamic `K_A`,
horizon-specific decoder, posterior network, or per-horizon noise is used.

## Fixed configuration

- WikiText-103 train/validation, BPE vocabulary 8192
- seed 1337; test split remains unread
- width 896, two reversible causal blocks
- exact inverse decoder, rms-tied head
- global `K=exp(A-A^T)`
- branch-code width 32; initializer bottleneck width 64
- initial branch residual RMS ratio 0.05
- horizons 4, trajectories 3, stride-one anchors 0 through 255
- posterior temperature 0.05
- h1 online attached relative-MSE weight 1
- prior KL weight 1
- effective batch 64, physical microbatch 16, four accumulations
- strict float32, TF32 disabled
- AdamW, clip norm 1.0
- 1000 updates on the 6000-update LR schedule
- validation starts: 128 examples from seed `1337+999`
- no stochastic trajectory draw after initialization; validation is exact

The physical microbatch differs from the recorded microbatch-32 controls
because the available GPU has 20 GiB. The objective has no cross-example
term, so all responsibilities, state losses, and prior losses remain
per-example and the effective batch remains 64.

## Reporting and success criteria

- reports at 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at 100, 250, 500, 750, and 1000
- select minimum validation posterior-weighted trajectory NLL
- branch-distance preservation error below `1e-5`
- nonzero branch residual RMS and token diversity at every horizon
- step-1000 clean h1 relative MSE at most `0.03`
- step-1000 validation marginal NLL at most the attached-h1-MSE control
  (`5.884858`)
- prefix-prior argmax agrees with the gold posterior winner above random
  chance (`1/3`)
- prior-selected mean h2--h4 accuracy at least the confidence-selected
  h1-MSE control mean (`0.0897`)

The last two criteria test train/inference alignment. Failure there does not
invalidate geometric commitment; it distinguishes persistent latent branch
identity from a useful prefix-conditioned trajectory prior.
