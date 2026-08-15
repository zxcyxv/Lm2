# 121M no-hidden-residual 500-step comparison

- Question: does removing only `z_next = rotated_z + delta` eliminate the
  persistent-write/memory-norm amplification mechanism by step 500?
- Comparison: the stopped 121M LR-2e-4 run; all registered settings stay fixed
  except `z_next = delta`.
- Seed/split: seed 1337, WikiText-103 train and fixed validation; test unread.
- Run: fresh initialization, H1--H4 CE, H16 monitor, batch 64, warmup 500,
  33,570-step LR schedule truncated at step 500.
- Required artifact: independent `step0500.pt` plus `last.pt`.
- Check: finite training, step-500 validation, then direct S/write alignment
  audit on `step0500.pt`.
