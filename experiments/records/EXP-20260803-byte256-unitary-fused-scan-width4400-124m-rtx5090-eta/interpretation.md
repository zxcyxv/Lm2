# Width-4400 / 124M RTX-5090 timing

Width 4400 is the closest architecture-valid multiple of 16 to 124M while
holding encoder depth and every non-width structural setting fixed. It gives
`123,941,528` total parameters and `122,810,728` trainable parameters.

## Canonical scheduled timing

The run used the ordinary 100-step linear learning-rate warmup for a 33,570
step schedule. Five optimizer steps initialized kernels and optimizer state;
the following fifteen full steps were timed.

| Quantity | Result |
|---|---:|
| Forward median | 271.601 ms |
| Backward + clipping median | 455.476 ms |
| Fused AdamW median | 2.580 ms |
| Complete optimizer step median | 729.976 ms |
| Peak allocated VRAM | 22.074 GiB |
| Supervised targets/s | 22,445 |

The fixed timing batch remained finite: loss decreased from `8.0355` on the
first timed step to `4.9581` on the last, and the final pre-clip gradient norm
was `105.256`. This short fixed-batch run establishes execution and timing,
not convergence or final language quality.

The width-1344 fused trainer's steady segment was `114.521 ms/step`, so width
4400 is `6.37x` slower while containing `9.38x` as many total parameters.
Batch 64 fits on the 32-GiB RTX 5090 without microbatching.

## Ten-epoch-equivalent ETA

The local train stream contains 55,000,000 byte tokens. Counting the 16
anchors by 16 horizons as 256 supervised target positions per sample gives
`16,384` targets per batch and `33,569` steps for ten epochs. At the measured
median this is `6.81 hours` of pure training.

If an epoch is instead defined by all 272 sampled bytes per window, the budget
is `31,595` steps or `6.41 hours`. The supervised-target definition is the
more conservative training estimate. Evaluation and checkpoint I/O should
put a practical single-GPU run near seven hours.

## Preserved conflicting smoke evidence

The initial `metrics.tsv` run mistakenly applied peak LR from its first step.
Its timing (`727.505 ms/step`) agrees with the scheduled measurement, but its
loss rose from `50.28` to `96.97`; it is retained only as timing evidence and
excluded from the training interpretation. The canonical evidence is
`metrics_warmup_schedule.tsv`.
