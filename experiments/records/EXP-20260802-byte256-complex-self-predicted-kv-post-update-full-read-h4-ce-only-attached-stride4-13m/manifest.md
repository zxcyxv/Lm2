# EXP-20260802 attached H4 CE with training stride 4

## Status

- State: completed; matched 1000-step run finished on 2026-08-02 UTC
- Matched stride-one control:
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h4-ce-only-13m/`
- Detached-gradient partial ablation:
  `../EXP-20260802-byte256-complex-self-predicted-kv-post-update-full-read-h4-ce-only-detached-cross-horizon-13m/`
- Test split remains unmaterialized and unread.

## Question

With the normalized attached-gradient H4 architecture unchanged, does reducing
training anchor overlap from stride 1 to stride 4 improve optimization or
memory efficiency while retaining four-step latent planning?

## Registered change

The sole mechanism/data-layout change from the attached control is:

```text
training anchor indices: 0, 4, 8, ..., 252
training anchors per sequence: 64
CE labels per sequence: 64 * 4 = 256
```

The control used 256 stride-one anchors and 1024 overlapping CE labels per
sequence. Central recurrence and causal decoder gradients remain fully
attached across all four horizons. There is no detach, MSE, KL, EMA, token
feedback, beta, independent head, or target model.

The transition remains post-update full-read with
`1/sqrt(write_count)` accumulated-read scaling. Four P states are decoded in
one exact-inverse causal tape and receive equal mean CE.

## Fixed protocol

- WikiText-103 raw UTF-8 train and validation bytes only
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed `1337 + 999`
- context 256; training anchor stride 4; validation anchor stride 4
- training and monitoring horizons: 4
- effective batch 64; physical microbatch 16
- strict float32; TF32 disabled
- AdamW, peak LR `3e-4`, clip norm 1.0
- 1000 updates on the fixed 6000-update schedule
- reports at 0, 1, 50, 100, then every 100 through 1000
- online model only

At equal update count and equal sampled windows, this run receives one quarter
as many overlapping CE labels as stride 1. It is intentionally not a
label-count-matched comparison. Unique input bytes, CE labels, wall time, and
peak VRAM are all reported so both views remain explicit.

## Comparisons and success criteria

Preflight must establish token-CE shape `[2,64,4]`, exact future-input
independence, fully attached nonzero H4 gradients to Q/K/V/readout/phases and
the online encoder, and the unchanged causal decoder triangle.

At step 1000:

- block and every per-horizon NLL must improve from initialization;
- block NLL below 3.0 is the minimum optimization criterion;
- primary comparisons are the full NLL curves against stride 1, not AR
  agreement;
- peak VRAM and wall time are compared with stride one's `7507823104` bytes
  and `1376.815886` seconds;
- a negative result is retained without replacing either producer.

Metrics are TSV and interpretation is Markdown. Checkpoints and smoke
artifacts remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_post_update_full_read_h4_attached_stride4_ce_only_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Evidence boundary

This isolates training anchor stride at H4. It does not test H128, stride 128,
sampled distribution equivalence, or the test split.

## Result

At step 1000, stride 4 reached block NLL `2.385889` with H1--H4 NLLs
`1.675063`, `2.269957`, `2.684619`, and `2.913915`. The stride-one control
reached block NLL `2.245606` at the same update count. Stride 4 used
`2816060416` peak bytes and `615.726084` seconds versus stride one's
`7507823104` bytes and `1376.815886` seconds.

The same-step NLL penalty was `0.140283`, while stride 4 was 2.24 times faster
and used 62.5% less peak allocated memory. At the nearest wall-time control,
stride-one step 500 used `692.5` seconds and reached block NLL `2.391656`, so
stride 4 reached a slightly lower NLL in less wall time despite receiving one
quarter as many overlapping labels per update.
