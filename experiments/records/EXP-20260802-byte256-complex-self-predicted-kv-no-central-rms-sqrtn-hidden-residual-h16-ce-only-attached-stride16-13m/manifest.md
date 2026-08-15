# EXP-20260802 no-central-RMS sqrt(count) hidden-residual H16

## Status

- State: registered but not launched; the exact control was interrupted after
  step 200 when the user redirected the architecture question to pre-norm
- Exact fresh control: `../EXP-20260802-byte256-complex-self-predicted-kv-no-central-rms-sqrtn-full-read-h16-ce-only-attached-stride16-13m/`
- Test split remains unmaterialized and unread.

## Question and sole intervention

Does a raw unitary hidden residual stabilize the no-central-RMS,
sqrt(write-count) H16 recurrence?

```text
control:   Z_{t+1} = full_read(Q(P_t), S_{t+1}) / sqrt(t+1)
candidate: Z_{t+1} = R Z_t + innovation_read(Q(P_t), W_t)
```

Everything else is identical: raw encoder-root copy, Identity central QKV
input, raw additive memory, `1/sqrt(write_count)` only on accumulated-memory
reads, no central post-normalization, H16 fully attached CE, and no auxiliary
loss, beta, detach, EMA, token feedback, or root reinjection.

## Fixed protocol

- Same WikiText-103 train/validation checksums and fixed starts as control
- seed 1337; validation seed 2336
- context 256, window 272, stride 16, H16 fully attached
- effective batch 64; microbatch 16
- strict float32, TF32 disabled
- AdamW, peak LR `3e-4`, clip 1.0, 6000-step schedule
- 300 updates; reports 0, 1, 50, 100, 200, 300
- 64 validation examples; evaluation microbatch 2

## Preflight, comparison, and success

Preflight must pass every control check and prove exact raw
`Znext = RZ + innovation_delta` with no normalization or parameter change.

Primary comparison is the complete fresh control curve. Stability success
requires all rows finite, raw pre-clip gnorm at most 10 at steps 50, 100, 200,
and 300, block NLL below 4 and improved from initialization, and every H1--H16
final NLL improved. The candidate must also report whether its step-300 gnorm
and block NLL beat the fresh control; neither value is known at registration.

Metrics are TSV, interpretation is Markdown, and checkpoints remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_no_central_rms_sqrtn_hidden_residual_h16_attached_stride16_ce_only_13m.py \
  --steps 300 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Evidence boundary

This isolates the hidden-successor residual inside the fresh no-central-RMS,
sqrt(count) architecture. It does not test write gates, `/t`, learned count
exponents, detach, or changes outside the central recurrence.
