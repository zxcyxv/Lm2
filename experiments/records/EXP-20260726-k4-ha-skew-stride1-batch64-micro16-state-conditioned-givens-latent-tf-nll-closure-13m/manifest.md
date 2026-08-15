# EXP-20260726 state-conditioned latent teacher forcing

## Status

- State: preregistered before optimizer update 1
- Authorization: user approved the architecture and requested implementation,
  documentation, and training
- Primary comparison:
  `../EXP-20260726-k4-ha-skew-stride1-batch64-micro16-selective-innovation-joint-nll-behavioral-closure-13m/`
- Long-run architectural reference:
  `../EXP-20260725-k4-ha-skew-stride1-batch64-micro16-selective-innovation-tape-three-trajectory-prior-tau005-h1-online-mse-6000step-13m/`
- Test split remains unread

## Question

Can one deterministic state-conditioned latent transition learn branch
commitment without latent coordinate regression, sampled candidates, or a
trajectory prior when:

1. every observed causal state supplies a canonical teacher-forced central
   input,
2. direct token NLL trains the next-token distribution, and
3. future behavioral KL transfers canonical branch-conditioned behavior to
   the self-fed open-loop latent rollout?

The experiment specifically tests whether the generated successor state can
serve as the branch commitment. It does not test calibrated sampling diversity
from one identical prefix.

## Fixed architecture

The reversible causal encoder, analytic exact-inverse decoder, token
embedding, RMS normalizations, causal attention, FFN layers, RoPE, and
RMS-tied vocabulary head are unchanged from the 13M latent-orbit family.

For every central input state `h`, one shared transition is used:

`T(h) = K(h) h + R(h)`.

Both terms are generated only from `h`. The transition receives no token ID,
token embedding, sampled noise, candidate index, trajectory prior, or global
plan.

### State-conditioned orthogonal operator

`RMSNorm(h)` is projected to a width-128 SiLU bottleneck. One zero-initialized
angle head emits two tapes of `width / 2` Givens angles.

The first stage rotates adjacent coordinate pairs. The second stage uses a
fixed one-coordinate cyclic pairing, applies another complete pairwise
rotation, and restores the original coordinate order. Consequently, for a
fixed conditioning state, both stages and their product are exactly linear
and orthogonal.

`K(h)` is identity at initialization because every angle is zero. The full
`width x width` matrix is never materialized.

### Selective innovation

The same bottleneck feature is projected directly back to model width:

`R(h) = W_R SiLU(W_f RMSNorm(h))`.

`W_R` is zero-initialized. There is no gate, direction normalization, learned
noise scale, or hard relative-scale bound. The observed relative RMS
`RMS(R(h)) / RMS(K(h)h)` is diagnostic only.

## Canonical latent teacher forcing

For an observed sequence:

`hA = F(x_<=A)`, `hB = F(x_<=B)`, and so on.

The teacher-forced predictions are:

`hB_hat_TF = T(hA)`,

`hC_hat_TF = T(hB)`,

`hD_hat_TF = T(hC)`,

`hE_hat_TF = T(hD)`.

The teacher supplies the canonical pre-central input. Every predicted
successor remains the model output.

The self-fed path starts from the same `hA`:

`hB_tilde = T(hA)`,

`hC_tilde = T(hB_tilde)`,

`hD_tilde = T(hC_tilde)`,

`hE_tilde = T(hD_tilde)`.

No token is decoded and re-embedded inside this central recurrence. The
completed open-loop latent tape is decoded causally in one batched readout.

## Objective

The primary token loss is ordinary teacher-forced NLL over the four observed
future tokens:

`L_NLL = mean_j CE(logits_TF_j, gold_j)`.

Behavioral closure begins at the second future decision. For horizons C, D,
and E:

`L_close_j = KL(stopgrad(p_TF_j) || p_OL_j)`.

The registered objective is:

`L = L_NLL + 1.0 * mean_j L_close_j`.

The following are excluded:

- latent MSE or latent NLL
- state or trajectory InfoNCE
- sampled noise
- multiple candidates or winner comparison
- candidate prior or prior KL
- token-conditioned central transition
- global latent plan

Teacher-forced and open-loop latent relative MSE remain diagnostics only.

## Fixed training configuration

- WikiText-103 train and validation splits; BPE vocabulary 8192
- seed 1337; validation starts use seed `1337 + 999`
- width 896, two reversible causal blocks
- exact inverse decoder and RMS-tied head
- transition bottleneck 128, two complete Givens stages
- four prediction horizons and stride-one anchors 0 through 255
- effective batch 64, physical microbatch 16, four accumulations
- strict float32, TF32 disabled
- AdamW, gradient clip norm 1.0
- 1000 optimizer updates on the 6000-update LR schedule
- reports at steps 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at steps 100, 250, 500, 750, and 1000
- validation examples: 128; test split remains unread

## Structural preflight criteria

- both state-conditioned Givens stages preserve arbitrary-vector norms and
  inner products to float32 tolerance
- the initialized transition is exactly identity to float32 tolerance
- the open first state and teacher-forced first state are identical
- changing future gold tokens does not change the self-fed latent tape or its
  logits
- teacher-forced NLL has finite nonzero gradients to the angle head,
  innovation head, encoder, and tied embedding
- behavioral closure has finite nonzero gradients through the self-fed
  transition and no gradient through its teacher probabilities
- closure excludes horizon 1 and is finite and nonnegative
- no forbidden candidate, prior, noise, or token-conditioning module is
  attached to the model

## Step-1000 success and failure criteria

Primary:

- validation teacher-forced token NLL is lower than its step-1 value
- validation open-loop token NLL is lower than its step-1 value
- validation behavioral-closure KL is lower than its step-1 value

Supporting:

- every optimization and validation metric remains finite
- open-loop top-1 accuracy is nonzero at all four horizons
- the learned Givens angles and innovation RMS ratio are finite and nonzero
- measured conditional-operator norm error remains below `1e-5`
- the teacher/open top-1 agreement curve does not collapse to zero after h1

The primary comparison record is descriptive rather than a strict numerical
gate because it uses a three-component trajectory mixture likelihood whereas
this experiment uses one teacher-forced conditional path.

## Evidence boundary

Passing these criteria establishes neither long-context sentence coherence
nor stochastic sample diversity. A later generation audit must compare the
self-fed latent rollout against literal-prefix re-anchoring on the same
generated history and report text separately from TSV metrics.
