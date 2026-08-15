# EXP-20260802 unitary branch-normalized residual H16

## Status

- State: completed; final stable regime reached, strict all-report criterion failed
- Structural control: `../EXP-20260802-byte256-complex-self-predicted-kv-unitary-block-residual-h16-ce-only-attached-stride16-13m/`
- Test split remains unread.

## Question

Does preserving raw unitary carriers while normalizing only the two coupling
branches stabilize fully attached H16 latent recursion?

```text
zbar_r = R z_r
zhat_r = zbar_r / rms(zbar_r)                    # fixed scalar RMS
q,k,v  = Q(zhat_r), K(zhat_r), V(zhat_r)
S_r    = U S_(r-1) + k v^dagger                 # raw carrier
m_r    = Re(q^dagger S_r) / (||S_r||_F + 1e-6)
z_(r+1)= zbar_r + O(m_r)                         # no scalar gate
decode = Dec(z_(r+1) / rms(z_(r+1)))             # branch only
```

`S_-1=0`. There is one write, one measurement, and one O call per iteration.
R and U are the existing unitary rotations. Gamma/damping is not implemented.
The QKV RMS has no learned affine, `/sqrt(r)` is absent, and neither carrier is
post-normalized. O uses ordinary linear initialization.

## Fixed protocol

- WikiText-103 train/validation bytes; test unread
- seed 1337; validation seed 2336
- context 256, H16, stride 16, window 272
- effective batch 64; microbatch 16
- float32, TF32 disabled
- AdamW, peak LR `3e-4`, clip 1.0, schedule 6000
- 300 updates; reports 0/1/50/100/200/300
- validation 64 examples, microbatch 2
- expected parameters 13,215,008; learned input-RMS affine removed

## Success and comparison

All rows must be finite. Raw pre-clip gradient norm must be at most 10 at
steps 50/100/200/300 and should approach the desired approximately 2-scale
regime. Final block NLL must be below 4, improve from initialization, and all
H1--H16 final NLLs must improve.

The structural control recorded gnorm `322.899536 / 60.648613 / 22.543360 /
7.396728` at steps 1/50/100/200 before interruption. This candidate must beat
it at matched available steps and remain below 10 through step 300.

Metrics are TSV, interpretation is Markdown, checkpoints remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m.py \
  --steps 300 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Evidence boundary

This jointly changes count scaling to instance Frobenius normalization,
removes the residual scalar, makes QKV RMS non-affine, and normalizes only the
decoder branch. It does not isolate those changes and does not test Gamma,
spectral constraints, detach, or carrier normalization.
