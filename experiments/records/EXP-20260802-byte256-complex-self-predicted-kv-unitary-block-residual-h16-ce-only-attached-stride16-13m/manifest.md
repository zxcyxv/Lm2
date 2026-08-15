# EXP-20260802 unitary SSM block-residual H16

## Status

- State: stopped by decision after step 50; stability criterion already failed
- Structural predecessor: `../EXP-20260802-byte256-complex-self-predicted-kv-input-prenorm-sqrtn-hidden-residual-h16-ce-only-attached-stride16-13m/`
- Test split remains unread.

## Question and intervention

Does closing the original single-layer unitary SSM at its block boundary give
stable fully differentiable H16 latent recursion without the predecessor's
separate preliminary and innovation-readout paths?

```text
Xbar_t = R X_t
Xhat_t = RMSNorm(Xbar_t)
W_t    = K(Xhat_t) V(Xhat_t)^dagger
S_t    = U S_(t-1) + W_t                 (S_0 is zero)
M_t    = read(Q(Xhat_t), S_t) / sqrt(t)
Delta  = O(M_t)                           (one O call)
X_(t+1)= Xbar_t + 0.1 Delta
CE state = recurrent successor = X_(t+1)
```

The output matrix uses its ordinary linear-layer initialization, not the
predecessor's `0.01/sqrt(read_width)` initialization. The residual scale is a
fixed `0.1`; it is not learned and adds no parameter.

## Fixed protocol

- WikiText-103 train and validation bytes only; seed 1337; validation seed 2336
- context 256, window 272, stride 16, H16 fully attached
- effective batch 64; microbatch 16
- float32; TF32 disabled
- AdamW, peak LR `3e-4`, clip 1.0, schedule 6000
- 300 updates; reports 0, 1, 50, 100, 200, 300
- 64 validation examples; evaluation microbatch 2
- expected parameters 13,216,352

## Checks, comparison, and success

Preflight must prove zero initial memory, exactly one write and one output
projection per step, exact block residual, identical decoder/recurrent state,
exact count sequence 0/1/2, sqrt(count) second read, ordinary output
initialization, and inherited causal/attached-gradient checks.

Compare against the structural predecessor gnorm at steps 1/50/100/200/300:
`247.010895 / 22.468979 / 191.509583 / 14.532938 / 17.269508`, with block
NLL `6.285072 / 3.452316 / 3.807190 / 3.267932 / 3.312959`.

Success requires finite rows, raw gnorm at most 10 at steps 50/100/200/300,
final block NLL below 4 and improved from initialization, every H1--H16 final
NLL improved, and lower step-300 gnorm than the predecessor.

Metrics are TSV, interpretation is Markdown, and checkpoints remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_unitary_block_residual_h16_attached_stride16_ce_only_13m.py \
  --steps 300 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Evidence boundary

This jointly corrects the closure topology and obsolete tiny output
initialization. It does not separately identify their effects and does not
test learned gates, `/t`, detach, post-norm, or forgetting.
