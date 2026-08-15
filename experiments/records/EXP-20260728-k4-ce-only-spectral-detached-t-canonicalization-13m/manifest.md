# EXP-20260728 CE-only spectral orbit with detached T canonicalization

## Status

- State: preregistered before optimizer update 1
- Authorization: user requested implementation and execution
- Primary comparison: paired raw `q_j = K_A^j h_A` versus
  `T(stopgrad(q_j)) stopgrad(q_j)` in the same fresh run
- Identity control: the T corrector is exactly identity-initialized
- Earlier checkpoints are not evidence for this architecture
- Test split remains unread

## Question

Can an input-conditioned spectral corrector reduce the state error between a
CE-only future proposal and the canonical encoder state obtained by actually
reinserting the proposal's own greedy token tape, while giving exactly zero
MSE gradient to the proposal, K, encoder, decoder, and token head?

This experiment tests detached latent canonicalization. It does not test the
earlier amplitude-shell hypothesis and does not claim greedy-AR equivalence.

## Base proposal architecture

The reversible encoder, analytic exact-inverse causal decoder, RMS-tied
embedding head, width 896, and two reversible blocks are unchanged.

For every causal anchor, one controller reads `stopgrad(h_A)` once. A global
orthogonal basis and prefix-conditioned complex eigenvalues define

`W = exp(S - S.T)`,

`K_A = W.T blockdiag_p(rho_A,p R(phi_A,p)) W`.

`K_A` is fixed for the four-token block. The raw proposals

`q_A,j = K_A^j h_A`, for `j = 1..4`,

are evaluated in closed form and decoded in one joint causal inverse tape.
There is no shell projection in the proposal path (`alpha=0`).

The base proposal receives only ordinary token cross entropy:

`L_CE = mean_j CE(Readout(q_A,j), gold_A+j)`.

There is no latent MSE, closure, KL, branch, prior, posterior, innovation, or
other auxiliary gradient into the base proposal.

## Self-selected canonical target

The proposal's greedy block is selected without gradient:

`b_A,j = argmax Readout(q_A,j)`.

During training only, the literal prefix and the selected four-token tape are
passed through the same encoder in one shared causal action-tape traversal:

`hstar_A,j = E(x_<=A, b_A,1:j)`.

The target is detached. It uses the model's selected tokens, not the corpus
future tokens. Consequently the auxiliary question is whether the selected
trajectory can be returned to its own canonical encoder states, independently
of whether the selected tokens match the corpus targets.

## Detached spectral T corrector

For each horizon independently, a width-128 controller reads
`q_detached = stopgrad(q_A,j)` and emits 448 bounded log-amplitude corrections
and 448 bounded phase corrections. In the same spectral basis as K,

`z = stopgrad(W) q_detached`,

`tau_p(q) = exp(s_p(q) + i delta_p(q))`,

`qhat_A,j = stopgrad(W).T [tau(q_detached) * z]`.

The scale and phase heads are zero-initialized, so `qhat == q` at
initialization up to floating-point roundoff. The corrector is a complex
diagonal linear map for fixed q, but `T(q)q` is input-dependent and therefore
nonlinear as a complete function.

The auxiliary objective is detached-target row-relative MSE:

`L_T = mean ||qhat_A,j - stopgrad(hstar_A,j)||^2`
`            / ||stopgrad(hstar_A,j)||^2`.

Both uses of q and the shared basis are detached in the T path. The canonical
action-tape encoder runs under `no_grad`. Therefore:

- `grad(L_T, T) != 0`
- `grad(L_T, E/K/W/D/Head/q) == 0`
- `grad(L_CE, E/K/W/D/Head/q) != 0`
- `grad(L_CE, T) == 0`

Base and T parameter sets are clipped separately and stepped by separate
AdamW optimizers so the T gradient norm cannot rescale the CE-only base
update.

## Parallelism and intended inference use

All `q_1..q_4`, token logits, and `T(q_j)q_j` states are vectorized over the
horizon. The real-token re-encoding exists only to construct training and
evaluation targets. It is absent at inference.

The intended later use is to take the corrected final state as the root of
the next parallel block. That free-generation claim is outside this first
experiment; first the state canonicalization must pass.

## Fixed configuration

- WikiText-103 train and validation; BPE vocabulary 8192
- seed 1337; validation starts use seed `1337 + 999`
- shell calibration is performed only to satisfy the reused alpha-zero
  spectral module contract; the shell is excluded from both optimizers and
  has no forward effect
- test split remains unread
- width 896; two reversible causal blocks
- exact inverse causal tape decoder; RMS-tied embedding head
- one global learned orthogonal basis
- 448 prefix-conditioned spectral phase and decay pairs
- spectral controller bottleneck 128
- T controller bottleneck 128
- T maximum absolute log-scale 2.0
- T maximum absolute phase correction pi radians
- T identity initialization
- four horizons; stride-one anchors 0 through 255
- effective batch 64; physical microbatch 8; eight accumulations
- strict float32; TF32 disabled
- two AdamW optimizers; separate clip norm 1.0
- `L_CE` base update and `L_T` corrector update every optimizer step
- 1000 updates on the unchanged 6000-update LR schedule
- validation examples 128
- reports at steps 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at steps 100, 250, 500, 750, and 1000

## Step-100 AR versus block-4 likelihood audit

The step-100 checkpoint is evaluated before using later checkpoints for this
comparison.

- AR reference: every validation token is predicted by the h1 proposal from
  its literal gold prefix, so K is recompiled at every token.
- Parallel block-4: anchors are spaced four tokens apart. At each block
  boundary the literal gold prefix is supplied once, and horizons 1--4 are
  predicted together from one block-frozen `K_A`. Gold tokens inside the
  block are targets only and are not fed back until the next block boundary.
- Both paths use the same validation starts and checkpoint.
- Report mean NLL, `exp(mean NLL)` perplexity, top-1 accuracy, horizon NLL,
  and the block-4 minus AR gap.

This is a block-boundary teacher-forcing comparison, not fully free-running
generation. It directly measures the likelihood cost of replacing four
token-level AR conditionings with one parallel block conditioning.

## Structural preflight criteria

- alpha-zero proposal states equal the unprojected `K_A^j h_A` states
- T is identity at initialization below relative error `1e-6`
- batched hard-action tape encoding matches literal prefix-plus-action
  re-encoding below `5e-5`
- changing future corpus tokens does not change q, selected actions,
  canonical self-selected targets, or corrected states
- finite nonzero CE gradients reach K/basis/controller, encoder, inverse
  decoder, and tied embedding
- CE gradient to every T parameter is exactly zero or absent
- finite nonzero T-MSE gradients reach T scale and phase heads
- T-MSE gradient to q and every base parameter is exactly zero or absent
- total base gradient equals the CE-only base gradient
- all states, logits, losses, and gradients are finite

## Step-1000 success criteria

Primary:

- validation corrected canonical-state relative MSE is lower than the paired
  raw-q canonical-state relative MSE
- corrected/raw MSE ratio is at most `0.75` overall
- corrected h4 canonical-state MSE is lower than raw h4 MSE

Supporting:

- corrected-state inverse decode preserves the proposal-selected token at
  least 95% of the time
- validation proposal token NLL is below its step-1 value
- all four proposal horizons retain nonzero top-1 accuracy
- all T scale/phase values and gradients remain finite

Failure of the 0.75 ratio gate means `q_j` alone does not contain enough
information for this bounded complex-diagonal T, optimization was
insufficient, or the moving CE-only geometry is too unstable. A dense
`d x d` hypernetwork is not an automatic escalation because only `T(q)q` is
identified. The first escalation is a frozen-base T phase or additional
conditioning on `h_A`.

## Evidence boundary

- A lower `L_T` does not improve the already selected first token.
- This experiment measures self-selected block-state closure, not corpus-gold
  latent prediction and not the original mean-amplitude shell mechanism.
- Greedy-AR parity, multi-block generation speed, and long-run stability
  require a separate inference producer after this gate passes.
- Metrics are written to TSV and interpretation to Markdown. Checkpoints and
  smoke artifacts remain under `outputs/` and are not added to Git.
