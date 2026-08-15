# EXP-20260807 121M state-dependent feedback, peak LR 2e-4

## Status

- State: training in progress
- Parent: `EXP-20260806-byte256-unitary-feedback-h4-stride4-h16-monitor-10epoch-width4352-121m`
- Sole intended intervention: peak learning rate `1e-4 -> 2e-4`
- Fresh seed-1337 initialization; train and fixed validation only; test unread

## Fixed protocol

- WikiText-103 byte train and validation splits
- context 256; H1--H4 equal attached token CE; anchor stride 4
- full H16 validation at stride 16; H5--H16 are diagnostics only
- width 4352; two reversible encoder blocks; exact inverse decoder
- eight complex heads; key dimension 16; value dimension 31
- state-dependent unitary branch-normalized feedback recurrence
- exactly 121,336,064 total and 120,217,600 trainable parameters
- effective/physical batch 64; no gradient accumulation
- fused AdamW, betas `(0.9, 0.95)`, zero weight decay, clip norm 1.0
- 500-step linear warmup to peak LR `2e-4`, then cosine decay
- 33,570 optimizer updates; strict float32 with TF32 disabled

No architecture, objective, seed, data order, horizon attachment, optimizer,
batch, warmup length, or schedule length is changed from the registered 121M
parent.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch_121m_lr2e4.py \
  --steps 33570 --schedule-steps 33570 --batch 64 --microbatch 64 \
  --eval-examples 64 --eval-microbatch 4
```

The initial live process was launched with an equivalent runtime override to
avoid leaving the GPU idle while this wrapper was materialized. Its `run.tsv`
records this experiment ID and the unchanged registered objective; the first
metric row records LR `4e-7`, exactly the first step of the schedule above.
