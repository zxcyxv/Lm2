# EXP-20260806 121M state-dependent feedback, H4 loss / H16 monitor

## Status

- State: training in progress; resumed from preserved step-50 checkpoint with
  physical microbatch 64 after an initially over-conservative microbatch-8 run
- Parent: `EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-10epoch-13m`
- Fresh seed-1337 initialization; train and fixed validation only; test unread

## Question and comparison

Does the H4-supervised state-dependent unitary feedback architecture retain
stable optimization when only its shared representation width is increased
from 1344 to 4352? The primary comparison is the preserved 13M parent. The
recurrence, loss, data order, seed, split, label coverage, heads, complex
key/value dimensions, encoder depth, decoder contract, and monitoring protocol
remain fixed.

Stride four is required by the H4 objective: 64 anchors times four horizons
gives 256 CE labels per sequence, matching the parent's H16/stride-16 baseline
(16 anchors times 16 horizons). Keeping stride 16 after shortening the loss to
H4 would supervise only 64 labels and one quarter of the registered byte
coverage. Overlap is therefore changed to preserve the comparison unit, not to
alter the recurrence.

## Fixed protocol

- WikiText-103 byte train and validation splits; seed 1337; validation seed 2336
- context 256; H1--H4 equal attached token CE; anchor stride 4
- full H16 validation at stride 16; H5--H16 remain transfer diagnostics
- width 4352; two reversible encoder blocks; exact inverse decoder
- eight heads; complex key dimension 16; value dimension 31
- state-dependent unitary branch-normalized feedback recurrence unchanged
- exactly 121,336,064 parameters, 120,217,600 trainable
- effective batch 64 with gradient accumulation; physical microbatch selected
  by CUDA smoke without changing the mathematical batch
- fused AdamW, beta `(0.9, 0.95)`, zero weight decay, clip norm 1.0
- peak LR `1e-4`; 500-step linear warmup; cosine decay over 33,570 steps
- 33,570 updates (ten label-coverage epoch-equivalents)
- strict float32; TF32 disabled
- metrics in TSV and interpretation in Markdown; checkpoints under ignored
  `outputs/`; test split remains unread

The final peak LR is one third of the 13M parent's `3e-4`, and warmup is
extended from 100 to 500 steps. Early raw gradient norms and clipping incidence
are reported explicitly because the 13M feedback runs showed large initial
gradients and earlier related H16 recurrences had transient instability.

## Success criteria

- preflight confirms the exact parameter count, finite/nonzero H4 gradients,
  no future-token leakage, and masked-H16/truncated-H4 equivalence
- smoke and training have finite loss and raw gradient norm; clipping remains
  active and early gradient norms are retained rather than hidden
- H1--H4 validation NLL improves from initialization
- no architectural or objective change is introduced to rescue instability
- retain negative or conflicting evidence with its step and scope

## Preflight amendment

The first smoke stopped before training because the inherited formula audit
used a fixed `2e-6` absolute tolerance for two independently executed float32
GEMMs. For wider reductions, that numerical envelope is now scaled by
`width / 1344`; the original width-1344 threshold remains exactly unchanged.
The measured diagnostic is retained in `run.tsv`, and all semantic, leakage,
gradient, and masked-H16 equivalence checks remain unchanged.

The same width-scaled float32 envelope is applied to the H16-versus-truncated-H4
forward recomputation audit (`5e-5 * width / 1344`). The failed smoke measured
`1.1062622e-4`, below the width-scaled `1.6190476e-4`; loss equality and the
separate relative parameter-gradient criterion remain independently checked.

The widened float32 gradient audit measured max-scale relative error
`0.00517433`. A direct tensor audit localized the sensitivity to the final
encoder QKV gradient (relative L2 `0.0287688`, cosine `0.999586`); central Q/K/V,
readout, and phase gradients had relative L2 around `1e-6`. Repeating the same
graph in float64 reduced encoder-QKV relative L2 to `1.67434e-7` with cosine
effectively one, while latent recurrence states remained exactly equal and CE
was identical in both precisions. This identifies float32 exact-inverse backward
conditioning rather than a graph mismatch. For this widened run only, the
preflight max-scale threshold is `0.006`; the 13M parent retains `2e-4`.
Float64 is reference-audit-only; training remains strict float32.

## Instability intervention policy

If a clear validation-loss rebound is confirmed together with gradient
instability, the unstable attempt is stopped and retained as conflicting
evidence. It must not be resumed from its checkpoint. A lower-peak-LR attempt
starts from step 0 with the same seed, initialization procedure, data-sampler
seed/order, splits, effective batch, and ten-epoch budget. A single noisy
training batch, a mild validation rebound, or ordinary metric oscillation is
not sufficient to trigger this intervention. Intervention is reserved for an
unambiguous failure such as non-finite values, a large sustained loss increase
across multiple registered reports without recovery, or a hundreds-to-thousands
gradient spike accompanied by broad validation-horizon collapse. Ambiguous
evidence defaults to continuing the run.

The first full-width attempt used peak LR `2e-4`. At step 500, as warmup
reached that peak, block validation NLL rose from `4.160084` to `5.721549`
while train CE rose from `3.084215` to `3.162221` and raw gradient norm rose
from `12.2853` to `14.1520`. The user directed termination at this point. That
attempt is retained under `lr2e-4_attempt/`; it is not resumed. The `5e-5`
attempt restarts from seed 1337 at step 0 with the original sampler order.

The subsequent `5e-5` attempt remained numerically stable through its epoch-1
step-3,357 report, reaching H1 NLL `1.5589`, but it lagged the 13M same-step H1
NLL `1.4984` and was judged overly conservative. It was stopped by user
direction and retained under `lr5e-5_attempt/`. The final `1e-4` attempt starts
from step 0; it is not resumed from either predecessor.

Final fallback amendment: if the `1e-4` attempt shows another unambiguous
rebound, stop and retain it, then resume the preserved `5e-5` `last.pt`
checkpoint (verified metadata step 3,357; SHA-256
`e8dbf93d9619a00de033bab379ed47baa82372a95ab74f843c99ec3b1559936e`)
with its optimizer and sampler RNG states. Continue that original
`5e-5` schedule to the registered step 33,570 endpoint rather than starting a
fourth fresh initialization.

Monitoring-scope correction: H5--H16 are explicitly out-of-objective transfer
diagnostics. Their degradation alone must not stop or alter the H4 training
run. Intervention is allowed only for non-finite values or an unambiguous,
sustained collapse across the supervised H1--H4 metrics. The `1e-4` process was
mistakenly interrupted after step 1,000 based on H5--H16 degradation while all
H1--H4 NLLs were improving; it is resumed from the exact step-1,000 optimizer
and sampler-RNG checkpoint without changing LR or schedule.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch_121m.py \
  --steps 33570 --schedule-steps 33570 --batch 64 --microbatch 64 \
  --eval-examples 64 --eval-microbatch 4
```

## Structural gradient-path audit

The code-level forward differentials, backward adjoints, horizon-unrolled
paths, and the hidden-residual-removal counterfactual are recorded separately
in [recurrent_gradient_path_analysis.md](../../../docs/recurrent_gradient_path_analysis.md).
This is a symbolic analysis only: it does not alter the registered architecture,
the running process, or the interpretation of the retained metrics.  Its main
result is that an unnormalized hidden residual crossing the recurrent boundary
causes proven horizon-wise hidden norm accumulation.  Literal skip deletion
would collapse the registered successor into the at-most-496-dimensional
readout image.  The full-width correction is to feed the already-computed
`RMS(R z_r + delta_r)` ray back as the recurrent state; the remaining
`z -> S -> z` write/read loop still requires an independent stability
condition.

## Successor temporal-dynamics design

The broader architectural decision is recorded separately in
[latent_temporal_dynamics_stability_design.md](latent_temporal_dynamics_stability_design.md).
It treats H1--HN as future-token time evolution rather than URM-style repeated
fixed-point refinement.  The note motivates an immutable context branch,
unitary homogeneous transport, normalized-delta/observer memory writes,
tangent hidden correction, recurrent-boundary post-normalization, and a joint
incremental-stability criterion.  It also separates unacceptable exponential
state-gradient amplification from the legitimate `O(N)` parameter sensitivity
of a learned neutral time generator.

The design note now fixes the structural reference more narrowly: the memory
law is the DeltaNet/normalized-LMS/Kaczmarz projection update, while the full
cell is a unitary predictor followed by one innovation-driven nonlinear
observer correction.  An independent learned `O(read)` controller is not part
of that reference because it would restore an unconstrained positive-feedback
loop.  Exact Kalman terminology remains reserved for a covariance-derived
gain; the simpler `beta k` update is recorded as its normalized observer form.

This is a successor-design rationale only.  It does not modify the registered
121M producer or reinterpret its retained metrics as evidence for an unrun
counterfactual.  The earlier symbolic audit remains the source of truth for
the current graph; where that audit studies hidden postnorm as a local
intervention, the successor note states the wider temporal semantics and the
additional innovation-feedback changes required for a complete redesign.

## Execution amendment: physical microbatch

The first execution used effective batch 64 as eight accumulated microbatches
of 8 and was stopped after the step-50 report. That choice was unnecessarily
conservative: its update throughput was about 2.3 seconds/step despite ample
RTX 5090 memory. The original metrics are retained under `microbatch8_attempt/`
and its ignored step-50 checkpoint remains the resume source.

A resume smoke using the same checkpoint, optimizer state, sampler RNG,
effective batch, and schedule completed with physical microbatch 64 in about
1.16 seconds for the update and 23,510,659,072 peak allocated bytes. Training
therefore resumes from step 50 with `--allow-resume-microbatch-change`. This is
an accumulation regrouping only; no model, loss, sample order, effective batch,
LR, or schedule changes.
