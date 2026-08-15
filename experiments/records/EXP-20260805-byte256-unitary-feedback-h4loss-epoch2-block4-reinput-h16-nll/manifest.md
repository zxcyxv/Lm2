# Epoch-2 feedback block-4 reinput H16 NLL

## Status

- State: completed teacher-forced audit; superseded for the requested
  greedy-self-reinput block metric
- Parent: preserved feedback H4-loss epoch-2 checkpoint at step 6,714
- Training and fixed validation records only; test unread

The registered run completed with 16,384 finite validation labels. Its
gold-block-4-reinput H1--H16 mean NLL was `2.137559340`; the epoch-matched
parallel-scan open-H16 mean was `2.912845770`.

User clarification after completion established that the intended metric
re-enters greedy model predictions inside each sixteen-token gold-boundary
block. This record re-enters gold tokens every four positions and therefore
must not be used as the requested comparison. It is retained as a separately
scoped teacher-forcing result.

## Question and comparison

When the H4-supervised state-dependent feedback model predicts sixteen
validation bytes as four consecutive four-byte blocks, how much does
canonically re-entering the gold four-byte block at each boundary change H16
block NLL?

Compare on the same epoch-2 budget:

1. current feedback open H16, with no token re-entry;
2. current feedback with gold block-4 re-entry;
3. the completed parallel-scan open-H16 run at step 6,714.

The parallel-scan reference has H1--H16 mean block NLL `2.912845770` and H16
single-horizon NLL `3.119666785`. These quantities must not be conflated.

## Fixed protocol

- checkpoint:
  `outputs/experiments/EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-10epoch-13m/step6714.pt`
- seed 1337; the parent's fixed 64 validation starts
- all 16 stride-16 anchors in each 256-byte context
- 1,024 prefix anchors and 1,024 labels per absolute horizon; 16,384 total
  labels for every aggregate mode
- at each anchor, predict local H1--H4, append the four gold bytes, re-encode
  the complete resulting prefix, and repeat for four blocks
- report every absolute H1--H16 NLL/accuracy, every four-byte block mean, and
  the H1--H16 aggregate
- strict float32, TF32 disabled; online checkpoint weights only
- metrics in TSV and interpretation in Markdown; no checkpoint mutation

Gold re-entry makes this a proper blockwise teacher-forced likelihood. It is
not the NLL of greedy generated-token re-entry and is not a free-generation
quality claim. Canonical prefix re-encoding also initializes the central
complex memory at each four-token boundary; it does not carry the preceding
rollout memory across that boundary while merely replacing the token state.
The parallel-scan reference remains open-H16, so the requested numerical
comparison is not a conditioning-matched architecture ablation.

## Success and evidence gates

- checkpoint step and 13,215,008-parameter architecture must match
- all 16,384 labels and all metrics must be finite
- block 1 must match the preserved feedback open-H1--H4 NLLs within `2e-3`.
  The first strict-float32 smoke observed max difference `9.38168e-4` because
  variable-length re-encoding and the preserved 272-token causal tape use
  different attention association shapes
- retain the result whether re-entry helps or hurts
- compare only the epoch-matched validation rows; test remains unread

## Producer

```bash
python eval_byte256_unitary_feedback_epoch2_block4_reinput_h16_nll.py \
  --microbatch 8
```
