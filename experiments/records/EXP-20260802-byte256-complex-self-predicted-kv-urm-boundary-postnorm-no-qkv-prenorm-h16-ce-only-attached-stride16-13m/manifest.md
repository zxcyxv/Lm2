# EXP-20260802 boundary post-norm H16 without QKV pre-norm

## Status

- State: running; one-update smoke preflight exposed a failed shell criterion
  before the registered execution
- Matched control:
  `../EXP-20260802-byte256-complex-self-predicted-kv-urm-boundary-postnorm-h16-ce-only-attached-stride16-13m/`
- Test split remains unmaterialized and unread.

## Question

Does removing only the shared learned RMSNorm immediately before the central
Q/K/V projections prevent the H16 divergence observed with final-carrier-only
boundary post-normalization?

## Sole intervention

```text
matched control: Q/K/V(h) = Linear(RMSNorm_learned(h))
this ablation:   Q/K/V(h) = Linear(h)
```

Everything else is held fixed. The initial `Z0` and `S0` remain raw. Inside
the transition, `P`, innovation, raw updated memory, and raw successor hidden
remain unnormalized. Only the complete successor `(Z_j, S_j)` receives the
same fixed non-affine boundary normalization before reuse. There is no
count-dependent read scaling, MSE, KL, EMA, detach, token feedback, beta, or
head change. Removing the learned RMSNorm removes its 1,344 affine parameters;
all remaining parameter initializations are unchanged.

## Fixed protocol

- WikiText-103 raw UTF-8 train and validation bytes only
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed `1337 + 999`
- context 256; required window 272
- training/evaluation anchor stride 16
- 16 anchors by 16 horizons = 256 CE labels per sequence
- effective batch 64; physical microbatch 16
- strict float32; TF32 disabled
- AdamW, peak LR `3e-4`, clip norm 1.0
- 1000 updates on the fixed 6000-update schedule
- reports at 0, 1, 50, 100, then every 100 through 1000
- 64 validation examples, evaluation microbatch 2
- online model only

## Preflight and metrics

Preflight must prove the Q/K/V input map has no normalization module or
trainable norm parameters and obeys the raw linear scale law. It must also
retain future-input independence, causal decoder triangularity, finite nonzero
H16 gradients to Q/K/V, complex readout, phases, and the online encoder, raw P,
and bounded reused successor Z/S.

### Preserved smoke conflict and execution amendment

The first exact one-change smoke attempt was rejected by the preregistered
near-unit successor-shell check: successor Z RMS was `0..0.000107534`, while
successor S remained bounded. No optimizer update occurred. This is retained
as evidence against the simple hypothesis, not erased by changing the model.

For the full registered run, the model computation remains unchanged. The
diagnostic gate is relaxed only enough to record finite, non-all-zero successor
Z values, so training can test whether the raw-linear transition leaves or
recovers from this initialization regime. Near-unit initial Z remains a failed
original success criterion in the final interpretation.

Metrics remain TSV and interpretation remains Markdown. They include H1--H16
NLL and accuracy, block NLL, raw global gradient norm, P/Z/memory and logit
scales, innovation energy, VRAM, and elapsed time. Checkpoints and smoke
artifacts remain ignored.

## Comparisons and success criteria

The matched control was stopped after step 300 with block NLL `223.148930`,
raw global gradient norm `604649216`, H1 P RMS approximately `28.01`, and H1
logit RMS approximately `373.72`.

This ablation must remain finite through 1000 updates, improve every horizon
NLL from initialization, and finish below block NLL `4.0`. Strong evidence for
the intervention additionally requires all registered step-300 scales to
remain finite and materially below the divergent control. Step-50 and step-100
gradient/NLL comparisons are retained even if they conflict with the final
result.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_urm_boundary_postnorm_no_qkv_prenorm_h16_attached_stride16_ce_only_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Evidence boundary

This isolates only removal of the central Q/K/V pre-RMSNorm. It does not test
normalizing the initial carrier, controlling innovation magnitude, changing
the decoder-facing P scale, H32/H128, AR equivalence, or test performance.
