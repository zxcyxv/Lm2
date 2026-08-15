# EXP-20260726 token-conditioned sequential transition

## Status

- State: preregistered before optimization update 1
- Authorization: user approved implementation after accepting sequential
  recurrence instead of associative parallel scan
- Primary comparison point: step 1000
- Primary architecture control:
  `../EXP-20260725-k4-ha-skew-stride1-batch64-micro16-selective-innovation-tape-three-trajectory-prior-tau005-h1-online-mse-13m/`
- Loss-design reference:
  `../EXP-20260726-k4-ha-skew-stride1-batch64-micro16-selective-innovation-joint-nll-behavioral-closure-13m/`
- The loss-design reference was interrupted after its recorded step-100
  evaluation; it is evidence at steps 1, 50, and 100, not a completed
  step-1000 control.
- Test split remains unread.

## Question

Can an observed-token-conditioned recurrent state remove latent candidate
assignment entirely while preserving the intended one-step language-model
semantics?

The previous selective-innovation experiments sample three latent paths and
compare their four-token costs. This experiment instead treats the observed
next token itself as the branch label. At training time the gold token updates
the successor state; at inference time the token sampled from the model must
be passed through the identical update.

This is a compound architecture/objective viability test, not an isolated
attribution experiment: candidate sampling, the prior, the innovation scan,
and path comparison are all removed together and replaced by one
token-conditioned recurrent transition.

## Sequential transition

Let `s_0 = hA`. For horizon `j`:

`z_j = K s_(j-1)`,

`p_j = Readout(prefix, s_1, ..., s_(j-1), z_j)`,

`s_j = U_theta(z_j, embedding(gold_j))`.

`z_j` predicts the current token before that token enters the recurrent
state. Therefore the current-token CE cannot leak through the gold token.
The conditioned state `s_j` is used only for later-token predictions.

`U_theta` is a bounded residual update:

`f_j = SiLU(W_z RMSNorm(z_j) + W_e RMSNorm(e_j))`,

`g_j = sigmoid(W_g f_j)`,

`raw_r_j = W_o[g_j * tanh(W_r f_j)]`,

`s_j = z_j + alpha_j * RMS(z_j) * unitRMS(raw_r_j)`.

`alpha_j` is produced from `f_j` and bounded to `[0.001, 0.25]`, initialized
at `0.05`. There is no ActNorm. The model keeps the existing embedding layer,
reversible causal encoder, exact inverse decoder, RMS-tied head, and global
orthogonal `K = exp(A-A.T)`.

The recurrence is evaluated in literal horizon order. No associative scan is
used. During training, readout tapes for all horizons share inverse-prefix
work, but that batching is an implementation optimization and is not a
parallel-generation claim.

## Readout without label leakage

Prediction tape `j` contains the already conditioned states
`s_1, ..., s_(j-1)` followed by the unconditioned proposal `z_j`.
Only the final diagonal slot is scored. For example:

```text
B: [z_B]
C: [s_B, z_C]
D: [s_B, s_C, z_D]
E: [s_B, s_C, s_D, z_E]
```

Thus `B` is predicted from `hA`; only `C` and later predictions can depend on
the supplied `B`.

## Objective

The primary objective is the single-path token negative log likelihood:

`L_CE = mean_j CE(p_j, gold_j)`.

This is the sampled-data form of forward KL at the conditional token
distribution. There is no mixture log-sum-exp, candidate responsibility,
winner reward, InfoNCE, latent MSE, Gaussian latent NLL, or candidate-prior
loss.

Behavioral closure compares later-token distributions from the recurrent
conditioned state with the corresponding online encoder state. A canonical
teacher tape replaces `s_1, ..., s_(j-1)` with
`hB, ..., h_(j-1)` and uses `K h_(j-1)` as its final proposal.
Teacher construction and teacher logits are stop-gradient.

For horizons 2 through 4:

`L_close_j = KL(stopgrad(p_canonical_j) || p_recurrent_j)`.

The total registered loss is

`L = L_CE + 1.0 * mean_j=2:4 L_close_j`.

The teacher distribution predicts the next unobserved token. It is not
`Head(D(hB))` used to reconstruct the already observed `B`.

## Fixed configuration

- WikiText-103 train/validation, BPE vocabulary 8192
- seed 1337; validation starts from seed `1337+999`
- test split remains unread
- width 896, two reversible causal blocks
- exact inverse decoder and RMS-tied embedding head
- global bias-free `K=exp(A-A.T)`
- four recurrent prediction horizons
- one realized path; zero latent candidates
- token-conditioned transition bottleneck width 128
- transition scale range `[0.001, 0.25]`, initialized at `0.05`
- behavioral-closure weight 1
- behavioral-closure horizons 2, 3, and 4
- stride-one anchors 0 through 255
- effective batch 64, physical microbatch 16, four accumulations
- strict float32, TF32 disabled
- AdamW, clip norm 1.0
- 1000 updates on the 6000-update LR schedule
- validation: 128 examples
- reports at 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at 100, 250, 500, 750, and 1000

## Structural preflight criteria

- prediction logits have shape `[batch, anchors, horizons, vocabulary]`
- recurrent states exactly match the registered sequential recurrence
- the horizon-1 prediction tape contains `z_B`, not the gold-conditioned
  `s_B`
- changing the supplied `B` while holding `hA` fixed leaves horizon-1 logits
  unchanged and changes the conditioned state used for horizon 2
- finite nonzero CE gradients reach global `K`, transition state/token maps,
  gate, output, scale, encoder, and tied embedding
- finite nonzero closure gradients reach the recurrent student
- closure teacher logits and online gold states receive no gradient
- no candidate, prior, sampled-noise, responsibility, or scan tensor exists
- `K` is identity/orthogonal within `1e-4` at initialization
- all logits, losses, states, and gradients are finite

## Step-1000 success criteria

Primary:

- validation single-path token NLL below `5.917968`, the completed
  selective-innovation architecture control's registered step-1000 marginal
  NLL

Supporting:

- validation behavioral-closure KL below its step-1 value
- validation horizon-2 token NLL is finite and lower than its step-1 value
- the token-conditioned state changes under an alternative token input by a
  finite nonzero amount
- transition scale remains finite, nonzero, and inside its registered bounds
- global `K` remains orthogonal within `1e-4`
- no non-finite training or validation metric

Speed and peak VRAM are reported but are not success criteria.

## Evidence boundary

Passing teacher-forced likelihood and closure does not establish free-running
coherence. A later generation audit must sample a token from `p_j` and feed
that exact sampled token to `U_theta`; feeding an argmax, gold token, or a
separately sampled token would test a different process.

Because this experiment deliberately gives up scan-parallel generation, it
cannot support claims about AR-equivalent parallel decoding. It tests whether
explicit local branch conditioning removes the need for multi-candidate path
comparison.
