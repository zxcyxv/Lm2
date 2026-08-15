# Epoch-3 state-conditioned sentence-quality audit

## Status

- State: completed
- Authorization: user requested immediate sentence-quality inspection
- Parent training continues independently
- Checkpoint: parent `best.pt`, required epoch 3 / update 876
- Comparison checkpoint: Grassmann v4 epoch 2 / update 584 on its registered
  20-epoch schedule; no Grassmann epoch-3 checkpoint is preserved
- Split: WikiText-2 validation only; test remains unread

## Question

At the current epoch-3 checkpoint, does the state-conditioned latent model
produce readable, non-collapsed continuations, and how much quality is lost
when four latent decisions are read out before re-anchoring? How does the
evidence compare with the nearest available early Grassmann checkpoint?

## Fixed protocol

- the same eight validation chunks selected by seed 1337:
  `555, 954, 430, 677, 157, 798, 752, 701`
- 64 prompt tokens and 64 generated tokens
- greedy decoding
- `grassmann_ar`: epoch-2 Grassmann greedy autoregressive decoding
- `block1`: re-encode the generated history after every token
- `block4`: generate four latent states with `T(h)=K(h)h+R(h)`, read all four
  tokens, then re-encode the four selected tokens
- no token ID or embedding enters the within-block central recurrence
- CUDA execution after the parent training process was stopped and its memory
  released at the user's request
- per-sample metrics in TSV; decoded evidence and interpretation in Markdown

## Execution result

- Grassmann epoch-2 AR collapsed on 6 of 8 samples, primarily to repeated
  variants of “the first time”.
- Current epoch-3 block-1 collapsed on 2 of 8 samples and had higher
  distinct-2, but frequently entered section-marker and generic-phrase loops.
- Current epoch-3 block-4 collapsed on 3 of 8 samples and showed severe
  period-4 repetition (`0.547917`) and immediate repetition (`0.404762`).
- The registered block-4 non-collapse criterion failed.
- One-shot batch-8 wall times were Grassmann AR `1.020s`, block-1 `0.689s`,
  and block-4 `0.145s`. These are preliminary latency evidence without
  warmup, not a final throughput benchmark.
- Full decoded evidence: `generation.md`
- Aggregate metrics: `metrics.tsv`
- Interpretation: `interpretation.md`

## Metrics and criteria

Report reference accuracy, distinct-1/2, immediate and period-4 repetition,
repeated 4-gram coverage, longest identical run, and collapse rate.

This early checkpoint is considered non-collapsed if block-4 aggregate
collapse rate is zero and immediate repetition is below 0.20. Grassmann is
one epoch and 292 updates earlier, so its decoded text is contextual evidence,
not an epoch-matched architecture ranking. No method is claimed to have final
sentence quality regardless of whether these criteria pass.
