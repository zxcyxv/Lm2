# EXP-20260722: Householder K + ActNorm + training-time Gaussian noise, joint CE+MSE, 13M

## Status and provenance

- State: preregistered before execution
- Authorization: user-requested exploration of adding stochasticity to `K`;
  user picked "simple noise regularizer" from a menu of options (flow
  matching, diffusion, discrete branch codebook, simple noise), explicitly
  flagging the known risk from the decoder-sensitivity investigation
- Producer: `train_query_k1_householder_actnorm_noise_mse_ce_joint_13m.py`
- Seed: 1337; scratch initialization
- Comparator: `EXP-20260722-query-k1-householder-actnorm-mse-ce-joint-13m`
  (identical architecture, no noise; 3000 steps: val NLL `4.3695`, accuracy
  `26.37%`, h2 direct-K^2 accuracy `0%` on 512 samples despite state cosine
  `0.9999`)

## Question

Given the same day's finding that the exact-inverse decoder is catastrophically
sensitive near the true future state `h_(t+1)` (a random perturbation of the
same relative magnitude as `K q`'s own error is amplified just as much as
`K q`'s actual error is), does training with explicit Gaussian noise injected
into `K`'s output make the decoder (and the token head reading it) more
robust to small deviations -- improving CE and/or the h2 direct-K^2
zero-shot extrapolation accuracy that is currently exactly `0%`? Or does it
instead destabilize CE training, given the decoder's demonstrated fragility
to any perturbation near this scale?

This is registered as a genuine two-sided test: the noted risk (CE could get
worse, not better) is treated as a real possible outcome, not a foregone
conclusion.

## Objective and architecture

### Aborted first attempt (Design A, coupled) -- stopped before results used

The first implementation added noise to `K`'s output and used that *same*
noisy value for both the CE-decode path and the MSE-comparison path
(`s' = s+noise` fed to both). This was manually stopped at step 250 (train
NLL/accuracy looked unremarkable, but the design itself was flawed, not just
the numbers) once we noticed: since `E[noise]=0`,
`E[||s'-h_(t+1)||^2] = ||s-h_(t+1)||^2 + Var(noise)`, and because the
injected noise's scale is proportional to `||s||`, the model could cheapen
`Var(noise)` -- and hence the whole MSE term -- simply by shrinking `||s||`,
with no requirement that `s`'s *direction* actually improve. That shortcut is
indistinguishable from genuine improvement in the logged MSE metric, so
Design A's numbers are not usable evidence either way and are not reported
here.

### Registered objective (Design B, decoupled)

Noise is injected only on the path that gets decoded for CE. The
MSE-comparison state (and every other diagnostic: `gold_state_cosine`,
`action_closure_*`) always uses the clean, noise-free state:

~~~text
s_(t+1)        = ActNorm(Householder(q_(t+1)))                    # clean
noise          = epsilon * ( ||s_(t+1)|| / sqrt(d) ) * z,  z ~ N(0, I_d)
s'_(t+1)       = s_(t+1) + noise                     (training only, CE path)
s'_(t+1)       = s_(t+1)                             (eval: no noise)
CE             = mean cross_entropy(head(decode([h_<=t, s'_(t+1)])), x_(t+1))
MSE            = mean_t || s_(t+1) - stop_gradient(h_(t+1)) ||^2 / ||h_(t+1)||^2
loss           = CE + MSE
~~~

`epsilon` is a fixed hyperparameter (not learned), set to `0.05` (perturb by
~5% of the state's own RMS-per-dimension scale) for this first run, chosen as
a moderate value relative to the ~3% latent-space error already observed
between `Kq` and the true state. A learnable `epsilon` was considered and
rejected for this first test: with no counter-pressure keeping it away from
0, the model could trivially minimize any noise-induced CE penalty by
shrinking `epsilon` to 0, which would silently defeat the purpose of the
experiment.

## Data and optimization

Same as the comparator: vocabulary 8192, width 896, 2 reversible encoder
blocks, `rms-tied` head, context 256, batch 64 (memory-driven, matching the
Householder siblings), peak LR `3e-4`, 100-step warmup, cosine decay,
gradient clip 1.0, strict FP32, TF32 disabled, seed 1337. **Steps: 1000**
first (quicker read before committing to 3000, unlike the comparator).

## Success and stopping

Preflight must additionally confirm noise reaches the CE logits (nonzero
effect versus a no-noise forward pass at the same inputs), that the
MSE-relevant state is bit-identical regardless of `inject_noise` (Design B
decoupling actually holds), and that eval is exactly disabled/deterministic
(bit-identical no-noise forward called twice). Same gates as the comparator
otherwise (orthogonality, identity-at-init, exact-inverse roundtrip,
finite/nonzero gradients). Primary comparison: CE/NLL/accuracy and h2
direct-K^2 accuracy against the noise-free comparator at matched steps. A
single run, single seed, single noise scale: this cannot establish whether
noise injection helps or hurts in general, only in this one matched setting
at `epsilon=0.05`.
