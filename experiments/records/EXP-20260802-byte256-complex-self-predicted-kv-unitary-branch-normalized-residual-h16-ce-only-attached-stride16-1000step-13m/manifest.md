# EXP-20260802 unitary branch-normalized residual H16 continuation to 1000

## Status

- State: completed
- Parent: `../EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-normalized-residual-h16-ce-only-attached-stride16-13m/`
- Resume source: parent `step0300.pt`
- Test split remains unread.

## Question

Does the same branch-normalized unitary recurrence continue stabilizing from
step 300 to step 1000, and do its checkpoint scaling/Jacobian assumptions move
toward their proposed asymptotic regime?

## Exact continuation protocol

- Restore model, AdamW state, and train data-generator state from step 300.
- Preserve seed 1337, batch 64, microbatch 16, H16 stride 16, validation starts,
  peak LR `3e-4`, original 100-step warmup history, clip 1.0, and schedule 6000.
- Continue updates 301--1000 without changing architecture, LR, warmup,
  normalization, loss, split, or optimizer.
- Report the resumed baseline at 300 and updates 400/500/600/700/800/900/1000.
- After completion, repeat the registered scaling/Jacobian audit at later
  checkpoints, including centered residual carrier
  `||z_r - R^r z_0||` in addition to raw latent norm.

## Success and evidence

All rows must remain finite. Raw pre-clip gnorm should remain in or approach
the approximately 2-scale regime without a late spike above 10. Final block
NLL and every H1--H16 NLL must improve relative to the resumed step-300
baseline. H16 directional Jacobian product gain should decrease from the
step-300 audit mean `6.1767`; equality to one is not assumed.

Metrics are TSV, interpretation is Markdown, and checkpoints remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2 \
  --resume-checkpoint outputs/experiments/EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-normalized-residual-h16-ce-only-attached-stride16-13m/step0300.pt \
  --record-dir experiments/records/EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-normalized-residual-h16-ce-only-attached-stride16-1000step-13m \
  --output-dir outputs/experiments/EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-normalized-residual-h16-ce-only-attached-stride16-1000step-13m
```

## Evidence boundary

This tests late optimization of the unchanged structure. It does not test a
lower peak LR, longer warmup, Gamma/damping, spectral constraints, or a longer
training-time recurrent horizon.
