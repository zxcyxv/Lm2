# EXP-20260722: Householder K + ActNorm + learned reparameterized noise, joint CE+MSE, 13M

## Status and provenance

- State: preregistered before execution
- Authorization: user-supplied design (attributed by the user to a
  labeled-speculation response from the external review prompt in
  `wiki/open-questions/residual-imbalance-decoder-sensitivity-review-prompt.md`),
  replacing the fixed-`epsilon` noise sibling
- Producer: `train_query_k1_householder_actnorm_learned_noise_mse_ce_joint_13m.py`
- Seed: 1337; scratch initialization
- Comparators: `EXP-20260722-query-k1-householder-actnorm-mse-ce-joint-13m`
  (no noise, 3000 steps), `EXP-20260722-query-k1-householder-actnorm-noise-mse-ce-joint-13m`
  (fixed `epsilon=0.05` noise, Design B decoupled; at step 500 this fixed-noise
  version was still behind the no-noise comparator on CE/NLL/accuracy)

## Question

The fixed-`epsilon` sibling used one global noise scale for every example and
every channel. This experiment instead lets the model predict its own
per-channel, per-example noise scale (a VAE-style reparameterization), with
no explicit regularization term or hyperparameter beyond initialization: CE's
own gradient is claimed (as **labeled speculation**, not proven) to push
`sigma` down when noise causes wrong predictions, while an unrelated,
separately-registered mechanism (the decoder's demonstrated catastrophic
sensitivity near `h_(t+1)`, established as fact earlier the same day) could
in principle punish `sigma` collapsing to near-0 if the model becomes
reliant on operating in a numerically fragile regime -- this experiment does
not attempt to verify that second, more speculative half of the claim; it
only tests the concrete, checkable part: does a learned per-channel noise
scale reach a different, more useful operating point than the fixed scalar
did?

## Objective and architecture

Identical to the fixed-noise sibling's Design B decoupling (MSE always
compares the clean state to `h_(t+1)`; only the CE-decode path sees a
perturbed state), but the perturbation's scale is now predicted by a small
learned linear map instead of being a fixed scalar:

~~~text
u              = ActNorm(Householder(q_(t+1)))                    # clean
u_hat          = u / (||u|| / sqrt(d))                 # per-example RMS-1 input
log_sigma      = W_sigma(u_hat) + b_sigma              # learned, per-channel
m              = exp(log_sigma) * (||u|| / sqrt(d)) * eps,   eps ~ N(0, I_d)
u'             = u + m                                  (training only)
u'             = u                                       (eval: no noise)
CE             = mean cross_entropy(head(decode([h_<=t, u'])), x_(t+1))
MSE            = mean_t || u - stop_gradient(h_(t+1)) ||^2 / ||h_(t+1)||^2
loss           = CE + MSE
~~~

`W_sigma` is a single `Linear(896, 896)` with weight initialized to `0` and
bias initialized to `log(0.05)`, so at step 0 every channel's noise scale is
exactly `0.05` (matching the fixed-`epsilon` sibling's value) independent of
input -- training is then free to move `log_sigma` per-channel and
per-example from that common starting point. `u` is RMS-normalized before
feeding `W_sigma` specifically so the predictor's own weight scale does not
have to compensate for `||u||`, which was observed to vary by nearly an
order of magnitude across sibling checkpoints (`~55` to `~680`).

**Implementation note on gradient paths**: `SigmaPredictor`'s input is fully
detached (`u.detach()`, both its norm and direction) before predicting
`log_sigma`, and the actual sampled noise's magnitude scale
(`||u||/sqrt(d)`) is likewise detached in `forward_with_actnorm`. This keeps
two paths clean: (1) `sigma_predictor`'s own weights are shaped by CE alone,
without the rest of the model being able to reshape `u` to indirectly game
its own predicted noise scale; (2) CE cannot cheapen the noise term by
shrinking `||u||` itself (the same shortcut Design A's coupling allowed for
MSE). The only path back into `K`/`ActNorm` is the direct `clean_states` term
inside `decode_states = clean_states + exp(log_sigma)*scale*eps`.

No entropy bonus, KL term, or explicit penalty on `log_sigma` is added --
per the user-supplied design, only CE's gradient shapes `W_sigma`. This is a
deliberate, registered risk: if there is no counter-pressure at all, `sigma`
could still drift to `0` and silently defeat the experiment exactly as a
freely-learned scalar epsilon was rejected earlier in the day for the same
reason. This run's `log_sigma` trajectory (mean, min, max) is logged
explicitly so this failure mode is visible rather than silent if it occurs.

## Data and optimization

Same as both noise siblings: vocabulary 8192, width 896, 2 reversible
encoder blocks, `rms-tied` head, context 256, batch 64, peak LR `3e-4`,
100-step warmup, cosine decay, gradient clip 1.0, strict FP32, TF32 disabled,
seed 1337, 1000 steps.

## Success and stopping

Preflight must additionally confirm: `log_sigma` equals `log(0.05)`
everywhere at init (so initial behavior matches the fixed-epsilon sibling
exactly), noise reaches the CE logits, the MSE-relevant state stays
decoupled from `inject_noise`, eval is deterministic, and `W_sigma` receives
a finite nonzero CE gradient. Primary comparison: CE/NLL/accuracy and the
`log_sigma` trajectory against both the no-noise and fixed-epsilon siblings
at matched steps. A single run, single seed: this cannot establish whether
learned per-channel noise is better than fixed noise in general, only in
this one matched setting.
