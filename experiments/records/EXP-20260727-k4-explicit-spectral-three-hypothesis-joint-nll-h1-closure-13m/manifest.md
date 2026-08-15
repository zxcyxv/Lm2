# EXP-20260727 explicit spectral three-hypothesis joint NLL

## Status

- State: preregistered before optimization update 1
- Authorization: user requested implementation and verification
- Primary comparison point: step 1000
- Primary architectural control:
  `../EXP-20260725-k4-ha-skew-stride1-batch64-micro16-initial-condition-three-trajectory-prior-tau005-h1-online-mse-13m/`
- Proper-likelihood reference:
  `../EXP-20260726-k4-ha-skew-stride1-batch64-micro16-selective-innovation-joint-nll-behavioral-closure-13m/`
- Test split remains unread.

## Question

Can a global bank of explicit learned temporal frequencies transport three
deterministic prefix-conditioned spectral hypotheses with one shared
orthogonal operator, while a probability-temperature-one joint path
likelihood learns their prefix-conditioned mixture weights without sampled
noise, detached winner training, or later innovations?

The experiment tests a finite-horizon spectral mixture. It does not claim
that three hypotheses exhaust the language continuation distribution or that
a four-token result establishes infinite-horizon spectral closure.

## Explicit spectral operator

The latent width is divided into adjacent real coordinate pairs. Pair `m`
has one learned angular frequency `omega_m`, in radians per token:

`K = direct_sum_m R(omega_m)`.

For complex coordinate `c_m = h_(2m) + i h_(2m+1)`:

`K^n c_m = exp(i n omega_m) c_m`.

There is no dense skew generator, matrix exponential, state-conditioned
operator, context-conditioned angle, or horizon-specific operator. Powers
are evaluated directly by multiplying each angle by the integer horizon.

The 448 frequencies use deterministic evenly spaced midpoint near-identity
initialization spanning `[-0.05, 0.05]` radians/token. The endpoints are not
duplicated. This breaks the all-identity mode symmetry while keeping every
initial rotation close to identity. No learned basis `Q` is added; the
reversible encoder and exact inverse decoder learn the native spectral-pair
coordinates.

## Deterministic spectral hypotheses

The existing three fixed learned branch codes and shared
prefix-conditioned initializer are retained:

`delta_i = G(stopgrad(hA), code_i)`,

`pi = softmax(P(stopgrad(hA)))`.

The residuals are centered so:

`sum_i stopgrad(pi_i) delta_i = 0`.

Each hypothesis is initialized once around the clean first transition:

`u_i,1 = K hA + delta_i`,

and receives no later correction:

`u_i,j = K^(j-1) u_i,1`.

There is no Gaussian draw, sampled candidate, per-horizon innovation,
token-conditioned recurrence, state-conditioned transition, or branch
reselection. All hypotheses and horizons are evaluated exactly.

## Objective

For hypothesis `i`, the complete observed four-token path cost is:

`C_i = sum_j CE(logits_i,j, gold_j)`.

The primary likelihood is the normalized finite-mixture joint NLL:

`L_joint = mean[-logsumexp_i(log pi_i - C_i)] / H`.

This uses probability temperature one. The exact posterior

`q_i = softmax_i(log pi_i - C_i)`

is detached only for diagnostics. There is no straight-through surrogate,
low-temperature soft minimum, separate prior KL, winner reward, latent
InfoNCE, or branch-state MSE.

The clean center retains attached online h1 spectral closure:

`L_h1 = relMSE(K hA, hB_online)`.

The registered total loss is:

`L = L_joint + 1.0 * L_h1`.

## Fixed configuration

- WikiText-103 train and validation, BPE vocabulary 8192
- seed 1337; validation starts use seed `1337 + 999`
- test split remains unread
- width 896, two reversible causal blocks
- exact inverse decoder and RMS-tied vocabulary head
- 448 explicit learned pair frequencies
- initial frequency range `[-0.05, 0.05]` radians/token
- three deterministic learned spectral hypotheses
- branch-code width 32; initializer bottleneck width 64
- initial residual RMS ratio 0.05
- four horizons; stride-one anchors 0 through 255
- joint likelihood temperature 1
- clean attached h1 relative-MSE weight 1
- effective batch 64; physical microbatch 16; four accumulations
- strict float32; TF32 disabled
- AdamW; clip norm 1.0
- 1000 updates on the unchanged 6000-update LR schedule
- validation examples 128
- reports at steps 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at steps 100, 250, 500, 750, and 1000

## Structural preflight criteria

- direct `K^n` phase evaluation matches literal recurrence below `1e-6`
- arbitrary-vector norm and inner-product error below `1e-5`
- represented dense weight is orthogonal below `1e-5`
- joint NLL matches explicit `-log(sum_i pi_i exp(-C_i)) / H`
  below `1e-6`
- posterior responsibilities sum to one and are detached only in their
  diagnostic view
- prior-weighted initial residual centroid below `1e-5`
- pairwise hypothesis distances are preserved below `1e-5`
- finite nonzero joint-NLL gradients reach frequencies, initializer output,
  initializer scale, prefix prior, encoder, inverse decoder, and tied head
- finite nonzero h1-closure gradients reach frequencies and the attached
  online h1 target
- h1 closure gives no gradient to online h2--h4 targets
- changing future gold tokens does not change spectral states or priors
- shared-prefix multi-hypothesis decode remains equivalent to literal
  prefix expansion

## Step-1000 criteria

Primary:

- validation temperature-one learned-prior joint NLL below the matched
  initial-condition control's temperature-one marginal NLL at the same
  checkpoint, to be measured from its preserved logits rather than inferred
  from the old tau-0.05 headline
- clean h1 relative MSE at most `0.03`

Supporting:

- learned-prior joint NLL no worse than uniform-prior joint NLL
- finite nonzero use of all three hypotheses
- finite nonzero residual scale and token diversity at every horizon
- no non-finite optimization or validation metric
- exact operator orthogonality below `1e-5`
- no single frequency plane accounts for more than `50%` of residual
  spectral energy in the registered post-training audit

## Required spectral audit

Mode-level numeric results are written to a separate TSV and interpretation
to Markdown. The audit reports:

- `omega`, effective period, and short root-of-unity proximity
- canonical-state, initial-residual, and decoder-sensitivity-weighted energy
- one-step amplitude drift, phase residual, and complex coherence
- horizon-wise complex transport residual
- `K^q` return scores for `q = 2, 3, 4, 8`
- logit/probability orbit spectrum and token period repetition
- held-out horizons 8, 16, and 32

Individual eigenvectors are not interpreted as semantic axes. Native pair
modes may be grouped by nearby frequency when degeneracy makes a subspace,
rather than a single coordinate pair, the stable object.

## Evidence boundary

A four-token likelihood improvement establishes neither sentence coherence,
calibrated open-ended sampling, nor exact equivalence to autoregressive
decoding. The first run isolates whether deterministic spectral hypotheses,
proper joint likelihood, and one global phase-advance operator form a viable
finite-horizon model.

Checkpoint and smoke artifacts remain under `outputs/` and are not added to
Git. Metrics remain TSV; interpretation remains Markdown.
