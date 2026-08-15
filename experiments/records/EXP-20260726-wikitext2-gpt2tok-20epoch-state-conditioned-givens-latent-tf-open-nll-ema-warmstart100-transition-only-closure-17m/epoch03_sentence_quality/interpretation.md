# Epoch-3 sentence-quality interpretation

## Aggregate result

| method | reference accuracy | distinct-2 | immediate repeat | period-4 repeat | repeated 4-gram coverage | collapse rate |
|---|---:|---:|---:|---:|---:|---:|
| Grassmann epoch-2 AR | 0.0332 | 0.1250 | 0.0556 | 0.2208 | 0.9426 | 0.750 |
| current epoch-3 block-1 | 0.0234 | 0.2758 | 0.2718 | 0.1812 | 0.7787 | 0.250 |
| current epoch-3 block-4 | 0.0391 | 0.2163 | 0.4048 | 0.5479 | 0.7889 | 0.375 |

All three early checkpoints have poor greedy sentence quality. The failure
modes differ.

Grassmann is the most globally collapsed: six of eight prompts converge on
the phrase “the first time”. Its low immediate-repeat rate is misleading
because the repeated unit is a longer phrase.

The current block-1 model is more prompt-sensitive and lexically varied. It
occasionally forms locally plausible clauses such as “The film was released
in the United States”, but then repeats generic clauses or WikiText section
markers. Its lower collapse rate and higher distinct-2 are real, but do not
make the continuations coherent.

Block-4 is materially worse than block-1. It produces patterns such as “the
film of the the film of” and repeated four-position section-marker cycles.
The period-4 metric `0.5479` confirms that this is the same phase-cycle class
of failure previously observed in block generation, not merely ordinary
word repetition.

## What the PPL result does and does not establish

The current model's epoch-3 H1 PPL `312.65` shows faster next-token
likelihood convergence than the available early Grassmann checkpoints.
It does not yet imply good free generation. Greedy rollout exposes
distribution shift and high-frequency attractors that validation H1 NLL does
not measure.

Behavioral closure through four trained horizons has also not eliminated
block-phase collapse at this early checkpoint. The H2--H4 Open objective can
improve held-out token likelihood while the argmax trajectory still enters a
small repetitive attractor.

## Preliminary speed evidence

For one un-warmed batch of eight 64-token continuations on the same GPU:

| method | wall seconds | approximate generated tokens/s |
|---|---:|---:|
| Grassmann epoch-2 AR | 1.020 | 502 |
| current epoch-3 block-1 | 0.689 | 743 |
| current epoch-3 block-4 | 0.145 | 3,527 |

Block-4 is about seven times faster than this Grassmann AR implementation,
which supports the deployment-throughput hypothesis. The timing is not a
final benchmark: checkpoints are epoch-unmatched, each method was timed once,
and kernel warmup/order was not controlled.

## Conclusion

The current architecture is promising in update efficiency and preliminary
block throughput, but epoch-3 sentence quality is not acceptable. Block-1 is
less globally collapsed than Grassmann epoch 2; block-4 still suffers a
strong phase-cycle failure. The decoded evidence should therefore be treated
as a useful early diagnosis, not a generation-quality win.
