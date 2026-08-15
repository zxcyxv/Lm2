# EXP-20260804 iterative feedback-scan audit at CFM step 3357

## Status

- State: completed on the registered 64 validation windows
- Read-only checkpoint audit; no optimization or parameter changes
- Fixed validation split only; test remains unread

## Question

Starting from the compiled time-varying scan, do one or two global feedback
refinements move the H16 latent and token predictions toward the exact
state-dependent feedback recurrence while retaining scan evaluation inside
each refinement?

The primary checkpoint is the epoch-1 conditional-ray-CFM model at step
3,357.  Its matched CE-only step-3,357 checkpoint controls for the auxiliary
loss, and the completed CE-only step-33,570 checkpoint tests whether the same
solver behavior persists in a mature scan-trained model.  This audit concerns
only their ordinary deterministic language-model paths; the flow time
conditioner, bridge source, and flow endpoint are not evaluated.

## Compared central rollouts

All arms share one encoder pass, the same exact-inverse decoder, token head,
parameters, validation windows, and strict float32 arithmetic.

1. `compiled_scan`: the registered time-varying scan.  Its Q/K/V forcing tape
   is the root orbit `c_h = R^h z_0`.
2. `rescan_1`: start from `compiled_scan`, rebuild the step-`h` forcing row as
   `R z_(h-1)` from that output tape, recompute the exact unitary memory prefix
   scan, and solve `z_h = R z_(h-1) + delta_h` with one latent affine scan.
3. `rescan_2`: repeat the same global refinement once more from `rescan_1`.
4. `sequential_feedback`: the literal H16 central recurrence in which every
   measurement-updated successor produces the next Q/K/V row.

For a supplied state tape `Z`, call the global refinement map above `T(Z)`.
Any fixed point of `T` obeys the literal sequential feedback recurrence.  The
finite rescan arms are solver iterates, not independently parameterized model
variants.  Damping is fixed to one and no stochastic value noise is used.

## Fixed protocol

- primary checkpoint:
  `outputs/experiments/EXP-20260804-byte256-unitary-conditional-ray-cfm-h16-10epoch-13m/step3357.pt`
- primary checkpoint SHA-256:
  `bdf3b93edc4cdbe60bd6bc73f6c16398c0643576de10d394bcac11d8e60b647b`
- matched CE-only step-3,357 checkpoint and SHA-256:
  `outputs/experiments/EXP-20260803-byte256-unitary-fused-triton-scan-h16-10epoch-13m-rtx5090/step3357.pt`,
  `0129c8c3289e5a3aa2fba5d92bc93df4e1a47aff45254fdf3033e79e8a781a12`
- mature CE-only step-33,570 checkpoint and SHA-256:
  `outputs/experiments/EXP-20260803-byte256-unitary-fused-triton-scan-h16-10epoch-13m-rtx5090/step33570.pt`,
  `ed97d7081a0760c50ddda5e24b0d767e804b07207dc9f54e3d7a986ba1cbec14`
- seed 1337; no evaluation-time stochastic model input
- WikiText-103 byte validation split and the 64 already registered validation
  starts; test unread
- prefix 256, H16, anchor stride 16, evaluation microbatch 4 unless memory
  measurement requires a smaller preregistered fallback
- fused Triton rotating-frame memory and latent scans
- RTX A4500; strict float32; TF32 disabled
- report block and per-horizon NLL/accuracy for every rollout
- report centered-logit RMS, `KL(p_feedback || p_variant)`, top-1 agreement,
  latent relative RMS and latent cosine against `sequential_feedback`
- report normalized fixed-point defect `||T(Z)-Z|| / ||Z||`
- time identical full encoder-central-decoder forward calls after warmup with
  CUDA synchronization; timing is descriptive and excludes checkpoint load
- metrics are written to TSV and interpretation to Markdown

## Registered interpretation criteria

- Under the causal Picard contract, `compiled_scan`, `rescan_1`, and
  `rescan_2` must respectively match `sequential_feedback` through H1, H2,
  and H3 within `2e-4` maximum absolute logit error.  The tolerance includes
  fused-scan float32 association-order error.
- Solver convergence is supported only if feedback logit KL, latent relative
  RMS, and fixed-point defect are non-increasing from `compiled_scan` through
  `rescan_1` and `rescan_2`; conflicting metrics remain reported.
- A language-model quality improvement requires lower paired validation NLL,
  not merely smaller latent error.  Because the checkpoint was trained for
  `compiled_scan`, movement toward feedback may improve or worsen NLL and is
  reported without deleting either result.
- Runtime is a quality/latency tradeoff, not a required pass.  Exact feedback
  equivalence is not claimed from two refinements unless the measured errors
  reach numerical tolerance.
- Conclusions are limited to this checkpoint, H16, and fixed validation set.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python eval_byte256_unitary_iterative_feedback_scan.py
```
