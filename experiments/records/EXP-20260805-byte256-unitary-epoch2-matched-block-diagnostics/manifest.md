# Epoch-2 matched sixteen-byte block diagnostics

## Status

- State: completed
- Fixed validation only; test unread

The rank/temperature diagnostics preserve the ordering scan > H4 > H1.
Scan was best at temperature 1, so no global margin rescaling was needed.

## Question and comparison

On the identical 1,024 gold-prefix boundaries and identical sixteen target
bytes, does the H1/H4/scan ordering survive diagnostics that are less sensitive
to logit scale? Evaluate their natural greedy chunks 1/4/16 without assuming
that any architecture is margin-driven.

## Fixed protocol and success

- Step 6,714, seed 1337, byte-identical 64 validation starts, 16 anchors and
  16 targets: 16,384 labels/model.
- Gold resets only at the sixteen-byte boundary; inside it only greedy outputs
  are re-entered and hidden/memory are reset on canonical re-encoding.
- Report CE, top-1/5/10, target rank/MRR, margins, entropy, space rate, exact
  block/prefix, and a neutral temperature grid `0.25..8`.
- Save the same twelve anchor-240 gold/generated sixteen-byte samples.
- Retain all outcomes; metrics/samples TSV, interpretation Markdown.

## Producer

```bash
python eval_byte256_unitary_epoch2_matched_block_diagnostics.py
```
