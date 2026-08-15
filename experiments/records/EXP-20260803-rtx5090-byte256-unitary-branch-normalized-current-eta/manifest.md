# EXP-20260803 RTX 5090 current sequential-H16 ETA

## Status

- State: registered; execution pending
- Fresh initialization; training and validation splits only; test unread
- Hardware: one NVIDIA GeForce RTX 5090 (32 GB)

## Question and comparison

What end-to-end optimizer-step throughput does the currently selected
`unitary-branch-normalized-residual` H16 trainer achieve on the RTX 5090, and
what 6000-step wall-clock ETA follows from it? The comparison reference is the
same selected packed-QKV/eager-memory trainer measured on the preceding GPU;
this record does not compare model quality.

## Fixed protocol

- seed 1337 and a fresh model initialization
- WikiText-103 byte training split; fixed seeded validation start
- effective batch 64, microbatch 64, no gradient accumulation
- H16, stride 16, strict float32 with TF32 disabled
- 200 optimizer steps under the ordinary 6000-step LR schedule
- report only at step 200; one validation example at start and finish
- include sampling, forward/backward, global clipping, fused AdamW, evaluation,
  and checkpoint production in the recorded end-to-end wall clock
- selected packed-QKV path; experimental compiled memory kernel disabled

## Success and scope

The run succeeds if all losses and raw gradient norms are finite and 200 steps
fit in 32 GB. Report total wall time, training-interval time, nominal sampled
byte throughput, peak allocated VRAM, and a linear 6000-step ETA. This short
fresh run measures execution speed only; its NLL is not a convergence result.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_unitary_branch_normalized_rtx5090_eta200.py \
  --steps 200 --schedule-steps 6000 --batch 64 --microbatch 64 \
  --eval-examples 1 --eval-microbatch 1 \
  --record-dir experiments/records/EXP-20260803-rtx5090-byte256-unitary-branch-normalized-current-eta \
  --output-dir outputs/experiments/EXP-20260803-rtx5090-byte256-unitary-branch-normalized-current-eta
```
