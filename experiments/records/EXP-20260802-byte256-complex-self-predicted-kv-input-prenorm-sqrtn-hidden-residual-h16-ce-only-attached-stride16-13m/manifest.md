# EXP-20260802 input-pre-norm sqrt(count) hidden-residual H16

## Status

- State: registered; implementation/preflight pending
- Exact control: `../EXP-20260802-byte256-complex-self-predicted-kv-post-update-full-read-h16-ce-only-attached-stride16-13m/`
- Test split remains unmaterialized and unread.

## Question and sole intervention

Does replacing the absolute full-read hidden successor with a raw unitary
residual stabilize the input-pre-norm, sqrt(write-count) H16 recurrence?

```text
shared:
  P_t       = read(Q(N(R Z_t)), U S_t) / sqrt(t)
  W_t       = K(N(P_t)) V(N(P_t))^dagger
  S_{t+1}   = U S_t + W_t

control:
  Z_{t+1}   = full_read(Q(N(P_t)), S_{t+1}) / sqrt(t+1)

candidate:
  Delta_t   = innovation_read(Q(N(P_t)), W_t)
  Z_{t+1}   = R Z_t + Delta_t
```

Only hidden arguments to Q/K/V use the shared learned RMSNorm. Memory, initial
root, and successor carriers remain raw. There is no recurrent post-norm.

## Fixed protocol

- WikiText-103 train and validation bytes only; test unread
- train SHA-256 `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256 `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed 2336
- context 256, window 272, stride 16, H16 fully attached
- effective batch 64; microbatch 16
- strict float32; TF32 disabled
- AdamW, peak LR `3e-4`, clip 1.0, unchanged 6000-step schedule
- 300 updates; reports 0, 1, 50, 100, 200, 300
- 64 validation examples; evaluation microbatch 2
- 13,216,352 total and 12,870,944 trainable parameters

## Preflight and metrics

Preflight must prove learned input RMSNorm is active, initial and recurrent
carriers are raw, accumulated-memory reads use exact `1/sqrt(write_count)`,
the successor is exactly `RZ + innovation_delta`, and the inherited causal,
inverse, attached-gradient, and parameter checks pass.

Metrics retain the matched TSV schema. Interpretation is Markdown and
checkpoints remain ignored.

## Comparison and success

The completed control prefix at steps 1/50/100/200/300 has raw gnorm
`699.308167 / 88.909645 / 125.927071 / 92.986031 / 1103.300659` and block NLL
`6.263249 / 3.548438 / 3.814659 / 4.839095 / 5.410936`.

Candidate success requires all rows finite, raw gnorm at most 10 at steps
50/100/200/300, block NLL below 4 and improved from initialization, and every
H1--H16 final NLL improved. Step-300 gnorm and block NLL must also beat the
exact control.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_input_prenorm_sqrtn_hidden_residual_h16_attached_stride16_ce_only_13m.py \
  --steps 300 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Evidence boundary

This isolates only the hidden successor construction inside the exact existing
input-pre-norm, sqrt(count), raw-carrier H16 architecture. It does not test
post-norm, initial-root normalization, `/t`, detach, or write gating.
