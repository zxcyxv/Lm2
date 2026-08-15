# EXP-20260802 step-6000 interpretability audit

## Status

- State: completed
- Source checkpoint: step 6000 of the microbatch-64 continuation
- Validation split only; test remains unread

## Question and comparison

Compared with the preserved step-1000 audit on the same eight fixed validation
windows, does six-times-longer training produce less collapsed H16 text and a
more selective, grounded sequence of measurement corrections?

## Fixed protocol

- seed 1337 + 999 and the same first eight fixed validation starts
- last stride-16 anchor at byte position 240; inspect the known next 16 bytes
- greedy H1--H16 decode, probabilities, exact accuracy and unique-byte count
- exact per-write measurement decomposition and interference effective rank
- correction, latent and memory norms by horizon
- verify measurement reconstruction and learned-R final-frame reconstruction
- additionally generate 128 bytes from four fixed validation prompts using
  (a) H1 with one-byte re-encoding after every prediction and (b) H16 with
  sixteen-byte block re-encoding; record both verbatim in `generation.tsv`

## Success criterion and evidence boundary

Relative to step 1000, exact byte accuracy and output diversity should improve
without losing the numerical decomposition checks. Decoded bytes ground a step
only through the shared decoder; they do not prove human-like reasoning.
