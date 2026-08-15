# EXP-20260803 13M fused-scan ten-epoch run

## Status

- State: registered; execution pending
- Parent: fused Triton rotating-frame benchmark
- Training and fixed validation splits only; test unread

## Question and comparison

Does the 13,215,008-parameter width-1344 model remain trainable and continue
improving over a ten-epoch-equivalent WikiText-103 byte-token budget when the
central time-varying recurrence uses the fused Triton rotating-frame scan?
Compare final and epoch-level H1/H16/block NLL with the preserved 1000-step
scan record and the earlier 6000-step feedback evidence. This run is not a
claim of state-dependent-feedback equivalence: the scan forcing tape remains
the registered state-independent time-varying variant.

## Fixed protocol

- seed 1337
- `data/wikitext103_bytes/train.bin`: 55,000,000 byte tokens
- random training windows with replacement; 33,570 optimizer steps represent
  ten epochs by 16,384 supervised targets per batch
- width 1344, two reversible encoder blocks, exact inverse decoder
- H16, anchor stride 16, batch/microbatch 64
- eight heads, complex key dimension 16, value dimension 31
- fused Triton rotating-frame memory and latent scans
- strict float32, TF32 disabled, fused AdamW, clip norm 1.0
- 100-step linear warmup followed by cosine decay over 33,570 steps
- fixed 64-example validation set; report at steps 100, 300, 1000 and every
  3,357 steps; test split unread
- save model, optimizer, and data-sampler RNG every 3,357 steps. The original
  final-checkpoint-only process was stopped after epoch 1; its partial metrics
  remain at the record root, while the checkpoint-enabled restart writes to
  `epoch-checkpoint-run/` so the earlier evidence is not overwritten.

## Success criteria

- no non-finite loss or gradient and no runtime failure
- block NLL remains below its step-1000 scan value and shows whether later
  epochs improve, plateau, or regress
- record all scalar metrics in TSV and conclusions separately in Markdown
- save epoch checkpoints under ignored outputs; do not add them to Git
