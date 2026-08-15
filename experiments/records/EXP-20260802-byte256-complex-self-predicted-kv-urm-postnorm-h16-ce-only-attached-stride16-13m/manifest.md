# EXP-20260802 URM-style post-normalized H16 CE

## Status

- State: completed at step 1000; retained as the intermediate-norm control
- Immediate failed H16 control:
  `../EXP-20260802-byte256-complex-self-predicted-kv-post-update-full-read-h16-ce-only-attached-stride16-13m/`
- Successful H4/stride-4 control:
  `../EXP-20260802-byte256-complex-self-predicted-kv-post-update-full-read-h4-ce-only-attached-stride4-13m/`
- Test split remains unmaterialized and unread.

The one-update smoke run was kept outside the repository. Preflight measured
initialized/updated memory RMS near `1.000`, P/Z RMS in `0.968..0.993`, exact
future-input independence, causal decoding, and finite nonzero gradients to
all registered central/encoder paths. Its large initial logit and gradient
scale is retained as a risk to monitor rather than silently changing the
registered mechanism.

The completed run reached block validation NLL `3.136612` with final raw
global gradient norm `10.558392`. Its interpretation and metrics remain in
this record; the result is not relabeled as a recurrence-boundary post-norm
experiment because P and memory were normalized before the complete successor
state had been formed.

## Question

Does applying URM-style post-normalization to the complete recurrent carrier
prevent the H16 self-predicted complex recurrence from developing the
activation and gradient instability seen without post-normalization?

## Registered computation

```text
S0 = ComplexRMSNorm(write(encoded_prefix))

for j in 1..16:
    Sminus = U S_(j-1)
    P_j = RMSNorm(O(Read(Q(Z_(j-1)), Sminus)))
    W_j = K(P_j) V(P_j)^dagger
    S_j = ComplexRMSNorm(Sminus + W_j)
    Z_j = RMSNorm(O(Read(Q(P_j), S_j)))

[P_1, ..., P_16]
    -> one exact-inverse causal decode
    -> sixteen parallel token-logit rows

loss = mean(CE_1, ..., CE_16)
```

No `1/sqrt(write_count)` or other count-dependent read scaling is present.
Hidden RMSNorms and complex-memory RMSNorm are fixed, non-affine operations
with epsilon `1e-6`. The central and decoder paths remain fully attached.
There is no MSE, KL, EMA, detach, token feedback, beta, independent head, or
future-token input.

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
- online model only

## Comparisons and success criteria

Preflight must establish finite nonzero H16 gradients to Q/K/V, complex
readout, phases, and online encoder; exact future-input independence; causal
decoder triangularity; and no auxiliary or detached target. It must also
establish RMS approximately one for initialized and updated complex memory,
all P states, and all recurrent Z states.

At step 1000 every horizon NLL must improve from initialization and block NLL
must be below 4.0. Gradient norm, memory energy, P RMS, Z RMS, and logit RMS
must remain finite through H16. The failed control reached block NLL
`4.839095` at step 200 and `5.410936` at step 300, with step-300 gradient norm
`1103.301`. The H4/stride-4 control reached block NLL `2.385889`.

Metrics are TSV and interpretation is Markdown. Checkpoints and smoke
artifacts remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_urm_postnorm_h16_attached_stride16_ce_only_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Evidence boundary

This is an H16 activation/optimization ablation. It does not establish H32 or
H128 stability, sampled block quality, AR equivalence, or test performance.
