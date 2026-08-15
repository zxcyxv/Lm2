# EXP-20260802 attached H16 CE with stride 16

## Status

- State: running; matched 1000-step run launched on 2026-08-02 UTC
- Immediate H4/stride-4 control:
  `../EXP-20260802-byte256-complex-self-predicted-kv-post-update-full-read-h4-ce-only-attached-stride4-13m/`
- Dense H4/stride-1 control:
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h4-ce-only-13m/`
- Test split remains unmaterialized and unread.

## Question

Can the fully attached normalized central recurrence scale from a four-token to
a sixteen-token non-overlapping block while retaining approximately the same
training-memory order and improving all sixteen future-token likelihoods?

## Registered computation

```text
training/evaluation anchor indices: 0, 16, 32, ..., 240
anchors per sequence: 16
horizons per anchor: 16
CE labels per sequence: 16 * 16 = 256

state_0 = initialize(encoded prefix)
for j in 1..16:
    P_j, state_j = attached_central_transition(state_(j-1))

[P_1, ..., P_16]
    -> one exact-inverse causal decode
    -> sixteen parallel logit rows

loss = mean(CE_1, ..., CE_16)
```

Cross-horizon gradients remain fully attached through both the central
recurrence and causal decoder tape. The transition remains post-update
full-read with `1/sqrt(write_count)` read scaling. There is no MSE, KL, EMA,
detach, token feedback, beta, independent head, or target model.

H4/stride-4 and H16/stride-16 both expose 256 anchor-horizon slots per
sequence. H16 nevertheless has four times the sequential central depth, a
longer branch-local causal attention axis, and a 272-byte training window
instead of 260.

## Fixed protocol

- WikiText-103 raw UTF-8 train and validation bytes only
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed `1337 + 999`
- context 256
- training anchor stride 16; evaluation anchor stride 16
- training and monitoring horizons: 16
- effective batch 64; physical microbatch 16
- 256 CE labels per sequence
- strict float32; TF32 disabled
- AdamW, peak LR `3e-4`, clip norm 1.0
- 1000 updates on the fixed 6000-update schedule
- reports at 0, 1, 50, 100, then every 100 through 1000
- online model only

The validation starts use the same seed as prior runs, but the required window
is 272 bytes and the non-overlapping anchor partition is stride 16. Exact
per-position equality with H4 validation rows is therefore not claimed.

## Comparisons and success criteria

Preflight must establish:

- token CE shape `[2,16,16]`;
- exact held-out-future independence of all central states and logits;
- finite nonzero last-horizon gradients to Q/K/V, complex readout, phases,
  and the online encoder;
- perturbing P16 cannot change logits H1--H15 but changes H16;
- no detach or auxiliary target is active.

At step 1000, report H1--H16 NLL and accuracy individually, their block mean,
peak VRAM, wall time, unique-input throughput, and CE-label throughput. Every
horizon NLL should improve from initialization; block NLL below 4.0 is the
minimum optimization criterion. H4/stride-4 reached block NLL `2.385889`,
peak VRAM `2816060416` bytes, and wall time `615.726084` seconds. These are
directional comparisons rather than required equality.

Metrics are TSV and interpretation is Markdown. Checkpoints and smoke
artifacts remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_post_update_full_read_h16_attached_stride16_ce_only_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Evidence boundary

This tests deterministic H16 block likelihood with a fixed non-overlapping
partition. It does not establish H32/H128 behavior, sampled joint dependence,
or test-split performance.
