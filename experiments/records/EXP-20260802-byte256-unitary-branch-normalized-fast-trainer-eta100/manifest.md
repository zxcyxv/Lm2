# EXP-20260802 fast trainer 100-step ETA benchmark

## Status

- State: completed
- Fresh initialization; training split; test unread

## Question and comparison

What is the end-to-end throughput and 6000-step ETA of the currently modified
CE-only fast training script at effective batch 64 and microbatch 64? Compare
with the completed old-path continuation throughput, while recognizing that
this short run starts from fresh weights and uses reduced evaluation.

## Fixed protocol

- seed 1337; H16 stride 16; 13.215M parameters
- 100 optimizer steps, schedule length 6000, batch=microbatch=64
- current prefix-only/read-state-only fast training objective
- report only at step 100; one fixed validation example at start and finish
- include sampling, forward/backward, clipping, fused AdamW, evaluation and
  final checkpoint in wall-clock; output remains an ignored benchmark artifact

## Success and scope

All steps and gradients must remain finite. Report measured wall time,
trainer-reported byte tokens/s, peak VRAM, and linear 6000-step ETA. A 100-step
fresh run is an ETA measurement, not a quality or stability experiment.
