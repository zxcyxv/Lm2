# Stochastic value-write and ray-velocity 300-step continuation

## Status

- State: completed
- Parent: completed 13M fused-scan epoch-10 checkpoint at step 33570
- Split: WikiText-103 byte train and fixed validation only; test unread

## Question and matched comparison

Does a causal process-noise tape in the rank-one memory write remain stable,
does CE alone suppress or retain its effect, and does a small detached
sample-path ray-velocity auxiliary improve shallow latent transport without
materially damaging token NLL?

All arms start from the identical `step33570.pt` model weights, use a fresh
identical AdamW optimizer, see the identical sampled training windows in the
identical order, and run for 300 optimizer steps:

1. `deterministic_ce`: noise scale 0, ray-flow weight 0.
2. `stochastic_ce`: effective per-step value-noise scale 0.05, ray-flow weight
   0.
3. `stochastic_ray_flow`: effective per-step value-noise scale 0.05 and
   H1--H4 ray-flow weight 0.01.

This is a controlled stochastic discrete velocity experiment.  It is not
registered as full continuous-time flow matching, a likelihood claim for the
latent path, or evidence that a single future-encoder row is a predictive
distribution.

## Fixed mechanism

For a pre-sampled circular complex noise tape `xi_r`, the value side of each
rank-one write is

```math
\tilde v_r=v_r+0.05\,\operatorname{rms}(v_r)\xi_r,
```

and memory remains

```math
S_r=US_{r-1}+k_r\tilde v_r^\dagger.
```

The nominal diffusion rate is `0.2` with `Delta tau = 1/16`, giving the
effective Euler--Maruyama scale `0.2 sqrt(1/16) = 0.05`.  The complete noise
tape is sampled before evaluation, so it is another time-varying additive
write and does not change the rotating-frame associative scan algebra.

The auxiliary uses the decoder-visible carrier ray.  With observed online
future-encoder samples `g_r` detached from the target side,

```math
d_r^\mathrm{pred}=\widehat z_r-R\widehat z_{r-1},\qquad
d_r^\mathrm{target}=\widehat g_r-R\widehat g_{r-1},
```

and reports/optimizes their target-relative squared error over H1--H4.  Token
CE remains the primary H1--H16 objective.

## Fixed protocol

- seed 1337; data-generator seed 1337; process-noise seed 25617
- checkpoint: `outputs/experiments/EXP-20260803-byte256-unitary-fused-triton-scan-h16-10epoch-13m-rtx5090/step33570.pt`
- model: 13,215,008 parameters, width 1344, two reversible blocks
- H16, anchor stride 16, effective batch/microbatch 64
- fused Triton rotating-frame scan, strict float32, TF32 disabled
- fresh fused AdamW per arm, beta `(0.9, 0.95)`, weight decay 0
- constant learning rate `3e-5`, raw global clip norm 1.0
- reports at continuation steps 0, 50, 100, and 300
- fixed 64-example validation starts; deterministic and fixed-noise validation
- save only smoke checkpoints under ignored `outputs/`; metric tables are TSV

## Success and interpretation criteria

- `value_noise_scale=0` must reproduce deterministic states and logits within
  numerical tolerance, and the noisy fused scan must pass the preserved scan
  tests.
- every arm must keep finite loss and gradients through step 300.
- report raw pre-clip gradient norm, deterministic NLL, stochastic NLL,
  noise-induced logit RMS/top-1 disagreement, and H1--H4 ray loss/cosine.
- useful first evidence requires the stochastic-ray-flow arm to reduce fixed
  noisy H1--H4 ray loss relative to stochastic CE while deterministic block
  NLL regresses by no more than 0.05 from its own step-0 value.
- if CE-only drives noise sensitivity down, retain that as evidence that
  unpaired process noise is treated as nuisance.  If evidence conflicts, keep
  both results and restrict the conclusion to this scale, checkpoint, and
  300-step continuation.

## Post-run metric scope correction

The original parent TSV's `noise_logit_rms` compared the full-window causal
encode used to obtain detached sample targets with the older prefix-only fast
encode.  Its non-zero scale-zero control showed that this mixed kernel-shape
evaluation differences with the requested process-noise effect.  The parent
TSV is preserved unchanged.  Noise-only conclusions use the separately
registered corrected audit that evaluates scale zero and scale 0.05 through
the identical full-encode path.
