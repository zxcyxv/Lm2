# EXP-20260803 width-4400 fused-scan RTX-5090 ETA

## Status

- State: scheduled timing complete
- Training split used only for fixed benchmark windows; validation/test unread

## Question and comparison

With encoder depth, H16 recurrence, heads, complex key/value dimensions,
stride 16, and batch 64 fixed, what is the optimizer-step time after increasing
only model width from 1344 to 4400? Width 4400 gives 123,941,528 total
parameters and is the closest valid multiple of 16 to 124M. Compare its fused
Triton rotating-frame speed with the preserved width-1344 measurement and use
the result to estimate a ten-epoch-equivalent WikiText-103 byte-token budget.

## Fixed protocol

- seed 1337; WikiText-103 byte train split; test unread
- strict float32, TF32 disabled; RTX 5090
- context 256, H16, anchor stride 16, batch/microbatch 64
- two reversible encoder blocks, exact inverse decoder
- eight complex heads, key dimension 16, value dimension 31
- fused Triton rotating-frame scan and fused AdamW
- five warm-up optimizer steps followed by fifteen timed optimizer steps
- fixed sampled training window during timing to isolate model compute
- canonical timing uses the ordinary 100-step linear LR warmup and a 33,570
  step schedule. The initial constant-peak-LR smoke is retained in
  `metrics.tsv`, but its rising loss excludes it from training interpretation;
  the scheduled rerun is written separately.

## Success criteria

- the full forward, CE backward, gradient clipping, and optimizer step fit in
  VRAM without microbatching
- losses and gradients remain finite during the timing interval
- report median/mean step time, component times, peak allocated VRAM, and both
  sampled-byte and supervised-target definitions of a ten-epoch ETA in TSV
- keep timing interpretation in Markdown and do not save a checkpoint
