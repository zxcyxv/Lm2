# EXP-20260728 input-conditioned spectral collapse

## Status

- State: preregistered before optimizer update 1
- Authorization: user requested implementation, structural verification, and
  execution
- Primary comparison: a fresh run of the identical producer with `alpha=0`;
  no earlier checkpoint is evidence for or against this architecture
- Architectural references:
  `../EXP-20260727-k4-explicit-spectral-three-hypothesis-joint-nll-h1-closure-13m/`
  and
  `../EXP-20260726-k4-ha-skew-stride1-batch64-micro16-prefix-conditioned-orthogonal-affine-scan-ce-closure-13m/`
- Test split remains unread.

## Question

Can a prefix-conditioned but block-frozen normal operator generate four
future latent states in closed form, while a hard positive amplitude-shell
retraction prevents radial blur from compounding and improves direct
four-token likelihood over the same input-conditioned operator without the
retraction?

This first run establishes structural viability and optimization behavior.
It does not claim equivalence to greedy autoregressive decoding.

## Architecture

The reversible encoder, analytic exact-inverse causal decoder, RMS-tied
embedding head, width 896, and two reversible blocks are unchanged.

For every causal anchor, one controller reads `stopgrad(hA)` once. A shared
RMSNorm and width-128 SiLU bottleneck emit 448 conditional phase offsets and
448 conditional decay offsets. One global learned orthogonal basis is

`W = exp(S - S.T)`.

For anchor `A`, the block-frozen operator is

`K_A = W.T blockdiag_p(rho_A,p R(phi_A,p)) W`.

`rho_A,p = exp(-softplus(nu_A,p))` is strictly contractive and
`phi_A,p` is a small base frequency plus a bounded prefix-conditioned offset.
`K_A` is normal for every fixed prefix. It is computed once per anchor and is
not recomputed from generated latent states or decoded tokens.

In the shared spectral coordinates, a positive learned shell `Abar_p` is
initialized from a deterministic training-prefix calibration batch. The
current encoder has no final fixed-norm contract, so no artificial
`sqrt(width)` shell normalization is imposed.

The registered run fixes `alpha=1`. For nonzero spectral coordinates:

`Pi(z)_p = Abar_p z_p / |z_p|`.

This is a nonlinear radial retraction onto a product of circles, not a
linear quantum-measurement projector. It preserves phase and resets radius.

The complete future is evaluated without a horizon loop:

`beta_A,p = (1 - alpha) rho_A,p`,

`r_A,j,p = beta_A,p^j |W hA|_p`
`          + alpha Abar_p (1 - beta_A,p^j) / (1 - beta_A,p)`,

`u_A,j,p = r_A,j,p exp(i(arg(W hA)_p + j phi_A,p))`.

All four states are transformed back through `W.T` and decoded as one causal
future tape behind the literal prefix. There are no branches, candidate
codes, priors, posterior responsibilities, sampled noise, innovations,
token-conditioned recurrences, or all-horizon latent targets.

## Objective

The token objective is ordinary direct parallel cross entropy:

`L_CE = mean_j CE(Readout(u_A,j), gold_A+j)`.

The clean unprojected first transition is

`v_A,1 = K_A hA`.

It receives a detached one-step vector relative-MSE target:

`L_MSE = mean ||v_A,1 - stopgrad(hB)||^2 / ||stopgrad(hB)||^2`.

The registered loss is exactly

`L = L_CE + 1.0 L_MSE`.

There is no phase/amplitude-separated regression, h2--h4 latent alignment,
KL, InfoNCE, energy loss, rank loss, or consistency loss.

## Fixed configuration

- WikiText-103 train and validation; BPE vocabulary 8192
- seed 1337; validation starts use seed `1337 + 999`
- shell calibration uses train seed `1337 + 9100`, 16 windows
- test split remains unread
- width 896; two reversible causal blocks
- exact inverse causal tape decoder; RMS-tied embedding head
- one global learned orthogonal basis, initialized to identity
- 448 prefix-conditioned spectral phase and decay pairs
- controller bottleneck 128
- initial base phase midpoint range `[-0.05, 0.05]` radians/token
- maximum conditional phase offset `0.10` radians/token
- initial base radius `0.90`
- maximum conditional decay-logit offset `2.0`
- hard shell retraction `alpha=1`
- four horizons; stride-one anchors 0 through 255
- one-step detached relative-MSE weight 1
- effective batch 64; physical microbatch 16; four accumulations
- strict float32; TF32 disabled
- AdamW; clip norm 1.0
- 1000 updates on the unchanged 6000-update LR schedule
- validation examples 128
- reports at steps 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at steps 100, 250, 500, 750, and 1000

## Structural preflight criteria

- vectorized closed form matches literal `Pi(K_A u)` recurrence below `1e-5`
- hard-projected radii match the positive learned shell below `1e-5`
- represented basis is identity at initialization and orthogonal below `1e-5`
- sampled fixed-prefix `K_A` is normal below `1e-5`
- conditional phase and radius tensors vary across distinct prefixes
- changing future gold tokens does not change rollout states or logits
- single shared-prefix decode matches a literal causal tape below `1e-4`
- finite nonzero CE gradients reach the basis, phase controller, shell,
  encoder, inverse decoder, and tied embedding
- finite nonzero one-step MSE gradients reach conditional phase and radius
- one-step MSE gives exactly zero gradient to the gold encoder target
- no branch, prior, responsibility, innovation, sampled-noise, or
  token-conditioned module is attached
- all states, logits, losses, controller outputs, and gradients are finite

## Step-1000 criteria

Primary:

- validation four-token NLL is lower than its step-1 value
- clean one-step validation relative MSE is at most `0.05`
- projected horizon-2 NLL is below the matched unprojected horizon-2 NLL

Supporting:

- nonzero prefix variation remains in both phase and radius
- all learned shell amplitudes remain finite and positive
- no non-finite optimization or validation metric
- basis orthogonality remains below `1e-5`
- top-1 accuracy remains nonzero at all four horizons

No projection benefit is claimed until the fresh `alpha=0` control is run
from the same initialization and data order. Numeric metrics are written to
TSV; interpretation is written separately to Markdown. Checkpoints and smoke
artifacts remain under `outputs/` and are not added to Git.

