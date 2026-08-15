# EXP-20260726 prefix-conditioned orthogonal affine scan

## Status

- State: preregistered before optimization update 1
- Authorization: user requested a parallel-scan comparison despite the
  expected loss of token-local rebranching
- Primary comparison point: step 1000
- Primary control:
  `../EXP-20260726-k4-ha-skew-stride1-batch64-micro16-token-conditioned-sequential-ce-closure-13m/`
- The primary control was still running when this manifest was written. Its
  completed step-100 evidence is in `metrics.tsv`; later comparisons must
  state the exact available checkpoint.
- Secondary completed architecture control:
  `../EXP-20260725-k4-ha-skew-stride1-batch64-micro16-selective-innovation-tape-three-trajectory-prior-tau005-h1-online-mse-13m/`
- Test split remains unread.

## Question

What likelihood and closure quality is lost when token-conditioned recurrent
branching is replaced by a genuinely open-loop, prefix-only operator tape
whose complete future can be evaluated by an exact associative scan?

The experiment also asks whether the central state rollout is faster at an
unseen horizon of 128 when evaluated by scan rather than by the identical
literal recurrence.

This is an architecture comparison, not a claim that the prefix-only tape is
AR-equivalent. All future operators are fixed before the first future token is
decoded, so generated tokens cannot change later dynamics.

## Why not scan full dense matrices

An affine scan over arbitrary dense maps,

`h_j = K_j h_(j-1) + r_j`,

is algebraically associative, but composing `K_2 K_1` costs `O(d^3)` and
stores `O(d^2)` values per horizon. That removes the intended speed advantage.

This experiment instead uses independent two-dimensional rotations in one
shared learned orthogonal basis. Composition then costs `O(d)` per scan
element.

## Prefix-only orthogonal affine tape

Let

`Q = exp(A-A.T)`

be one learned global orthogonal basis and

`y_0 = Q.T hA`.

A horizon-shared planner receives only `hA` and an extrapolatable sinusoidal
encoding of position `j`. It produces:

- one angle `theta_j,k` for each adjacent two-dimensional pair `k`
- one innovation `rho_j` in the shared rotation basis
- one bounded relative-RMS innovation scale

The local recurrence is

`y_j = R(theta_j) y_(j-1) + rho_j`,

`u_j = Q y_j`.

Every `R(theta_j)` is block-diagonal with `2 x 2` rotation blocks and is
therefore exactly orthogonal. In the original latent coordinates the linear
operator is

`K_j = Q R(theta_j) Q.T`.

All `K_j` share the same basis and commute, but their angles and innovations
can vary with prefix and horizon.

The planner is:

`f_j = SiLU(W_h RMSNorm(hA) + W_p FourierPosition(j))`,

`theta_j = base_theta + 0.25 * tanh(W_theta f_j)`,

`raw_rho_j = W_r f_j`,

`rho_j = alpha_j * RMS(hA) * unitRMS(raw_rho_j)`.

`alpha_j` is bounded to `[0.001, 0.25]` and initialized at `0.05`.
The bottleneck width is 128 and the position feature width is 32. There are
no horizon-specific parameters, learned position lookup tables, sampled
noise values, candidates, or token inputs. There is no ActNorm.

## Exact associative scan

Represent one local affine map by `(theta, rho)`. Composition in chronological
order is

`(theta_2, rho_2) o (theta_1, rho_1)`

`= (theta_2 + theta_1, R(theta_2) rho_1 + rho_2)`.

This operation is associative. A Hillis--Steele prefix scan returns the
composed map for every horizon in logarithmic dependency depth. The states
are then

`y_j = R(Theta_1:j) y_0 + Rho_1:j`.

The production forward must use this scan. A literal sequential recurrence
exists only as a numerical reference and central-rollout timing control.

## Objective

The open-loop scan states are decoded jointly behind the real prefix. The
primary objective is direct token negative log likelihood:

`L_CE = mean_j CE(Readout(u_j), gold_j)`.

For behavioral closure, only the first state is replaced by the observed
online encoder state `hB`. The already compiled prefix-only suffix operators
and innovations for horizons 2 through 4 are reused without change. The
reanchored path is a stop-gradient teacher.

For horizons 2 through 4:

`L_close_j = KL(stopgrad(p_reanchored_j) || p_open_j)`.

The total objective is

`L = L_CE + 1.0 * mean_j=2:4 L_close_j`.

There is no latent MSE, InfoNCE, Gaussian NLL, candidate mixture, responsibility,
prior, winner reward, gold-token-conditioned transition, or low-temperature
soft minimum. Latent relative MSE remains diagnostic only.

## Fixed configuration

- WikiText-103 train/validation, BPE vocabulary 8192
- seed 1337; validation starts from seed `1337+999`
- test split remains unread
- width 896, two reversible causal blocks
- exact inverse decoder and RMS-tied embedding head
- one global learned orthogonal basis `Q=exp(A-A.T)`
- pairwise rotation coordinates: 448 adjacent pairs
- prefix/position planner bottleneck 128
- sinusoidal position feature width 32
- maximum context-conditioned angle delta `0.25` radians per step
- innovation scale range `[0.001, 0.25]`, initialized at `0.05`
- one deterministic open-loop path
- four trained horizons
- behavioral-closure weight 1, horizons 2 through 4
- stride-one anchors 0 through 255
- effective batch 64, physical microbatch 16, four accumulations
- strict float32, TF32 disabled
- AdamW, clip norm 1.0
- 1000 updates on the 6000-update LR schedule
- validation: 128 examples
- reports at 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at 100, 250, 500, 750, and 1000

## Structural preflight criteria

- scan states match the literal recurrence below `1e-5`
- scan composition matches explicit affine-map composition below `1e-6`
- each local rotation preserves vector norm below relative error `1e-6`
- `Q` is identity/orthogonal within `1e-4` at initialization
- changing future gold tokens while preserving the literal prefix does not
  change open-loop states or logits
- finite nonzero CE gradients reach `Q`, planner context/position maps,
  angle head, innovation head, innovation scale, encoder, and tied embedding
- finite nonzero closure gradients reach the open-loop scan
- reanchored teacher and online `hB` receive no closure gradient
- no candidate, prior, responsibility, sampled-noise, or token-conditioned
  transition tensor exists
- all losses, logits, states, scales, angles, and gradients are finite

## Step-1000 criteria

Likelihood/closure:

- validation direct token NLL at most `6.10`
- behavioral-closure KL below its step-1 value
- finite nonzero angle magnitude and innovation scale
- no non-finite metric

Parallel mechanism:

- at horizon 128, the central scan exactly matches the sequential reference
  below `1e-4`
- after CUDA warmup and synchronization, median central scan latency is lower
  than the median literal-recurrence latency for the registered benchmark
  shape `[batch=1, anchors=256, horizon=128, width=896]`

The horizon-128 latency test measures the central compiled state rollout only.
It does not include inverse decoding, the vocabulary head, sampling, or
end-to-end generation and must not be reported as end-to-end token speed.

## Evidence boundary

This model cannot respond to tokens generated inside its open-loop block.
Passing the scan and speed criteria shows only that a prefix-conditioned
future tape can trade conditional expressivity for parallel central rollout.

Free-generation evaluation must separately report quality and latency at
block lengths 4, 16, 32, 64, and 128. It must not describe the prefix-only
result as AR-equivalent, even if short-horizon likelihood matches the
token-conditioned sequential control.
