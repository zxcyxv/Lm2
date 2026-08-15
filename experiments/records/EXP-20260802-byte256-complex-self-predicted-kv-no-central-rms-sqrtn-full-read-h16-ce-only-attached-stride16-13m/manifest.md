# EXP-20260802 no-central-RMS sqrt(count) full-read H16 control

## Status

- State: interrupted by user-directed architecture change after the complete
  step-200 report; partial evidence and `last.pt` are preserved
- Candidate: `../EXP-20260802-byte256-complex-self-predicted-kv-no-central-rms-sqrtn-hidden-residual-h16-ce-only-attached-stride16-13m/`
- Test split remains unmaterialized and unread.

## Question and computation

This fresh control establishes the exact no-residual comparison for a central
recurrence with no central RMS normalization and deterministic accumulated-read
scaling.

```text
P_t       = read(Q(R Z_t), U S_t) / sqrt(t)
W_t       = write(P_t)
S_{t+1}   = U S_t + W_t
Z_{t+1}   = full_read(Q(P_t), S_{t+1}) / sqrt(t+1)
```

Central QKV pre-normalization, private initial-root RMSNorm, and all recurrent
hidden/memory post-normalization are disabled. The encoder/decoder architecture
outside the central recurrence is unchanged.

## Fixed protocol

- WikiText-103 train/validation bytes; test unread
- train SHA-256 `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256 `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed 2336
- context 256, window 272, stride 16, H16 fully attached
- effective batch 64; microbatch 16
- strict float32, TF32 disabled
- AdamW, peak LR `3e-4`, clip 1.0, unchanged 6000-step schedule
- 300 updates; reports 0, 1, 50, 100, 200, 300
- 64 validation examples; evaluation microbatch 2

## Preflight and success

Preflight must prove raw root/S/Z carriers, exact `1/sqrt(write_count)` second
read, Identity central QKV input, no central RMS flags, causal independence,
exact inverse triangularity, finite attached H16 gradients, and unchanged
13,215,008 parameters.

Success requires all rows finite, block NLL below 4 and improved from
initialization, and every H1--H16 final NLL improved. This control has no
gradient-ceiling success requirement; it supplies the exact residual ablation.

Metrics are TSV, interpretation is Markdown, and checkpoints remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_no_central_rms_sqrtn_full_read_h16_attached_stride16_ce_only_13m.py \
  --steps 300 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Evidence boundary

This control is fresh because older no-postnorm checkpoints retained central
QKV RMSNorm and therefore are not used as matched evidence.

## Partial result

The run remained finite through its last complete row. Raw pre-clip gnorm at
steps 1/50/100/200 was `286.947784 / 61.999916 / 27.511654 / 2.292760`;
block NLL was `6.168354 / 3.250937 / 3.401100 / 3.141457`. It was stopped
during the next training interval and is not a completed 300-step control.
