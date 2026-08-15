# EXP-20260803 feedback H16 RTX-5090 1000-step timing confirmation

## Status

- State: registered; execution pending
- Fresh initialization; train/validation only; test unread

## Question and comparison

Directly time 1000 optimizer steps of the unchanged state-dependent feedback
recurrence on the RTX 5090. This confirms the preceding 200-step ETA record by
an observed long run and provides the direct comparison requested after the
time-varying scan run.

## Fixed protocol

- seed 1337; WikiText-103 byte train split
- H16 stride 16; effective batch = microbatch = 64
- packed QKV, eager sequential memory update, experimental fusion disabled
- strict float32, TF32 disabled
- unchanged AdamW, clipping, and 6000-step LR schedule
- 1000 steps; report only at step 1000
- one fixed validation example at start/end to minimize timing contamination

## Success

All losses and gradients must stay finite. Report observed wall time,
seconds/step, nominal sampled-byte throughput, and peak VRAM. This record is a
timing confirmation rather than a statistically stable validation estimate.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_unitary_branch_normalized_rtx5090_1000_confirm.py \
  --steps 1000 --schedule-steps 6000 --batch 64 --microbatch 64 \
  --eval-examples 1 --eval-microbatch 1 \
  --record-dir experiments/records/EXP-20260803-byte256-unitary-feedback-h16-1000step-rtx5090-confirm \
  --output-dir outputs/experiments/EXP-20260803-byte256-unitary-feedback-h16-1000step-rtx5090-confirm
```
