# Evaluation interpretation

The fixed step-1000 checkpoint failed both the one-step canonicalization and
the recurrent alternating K--T contracts.

Raw greedy AR and alternating K--T agreed on `0.833496` of h1 tokens. This is
close to the parent training evaluation's one-step corrected/teacher top-1
agreement of `0.842`, so the independent rollout producer reproduces the
same initial correction error.

Agreement then fell to `0.111816`, `0.082764`, and `0.124268` at horizons
2--4. Overall token agreement was `0.288086`, and exact four-token block
agreement was `0.006958`. Both preregistered equivalence gates (`0.90` token,
`0.75` exact block) failed.

The NLL diagnostic also separated the paths:

| policy | NLL | PPL-like | accuracy |
|---|---:|---:|---:|
| raw self-fed AR | 7.868222 | 2612.92 | 0.0475 |
| alternating K--T | 11.078192 | 64743.70 | 0.0507 |
| projected alternating K--T | 10.757650 | 46988.10 | 0.0350 |

Alternating K--T h1 NLL was already `9.099708`, versus raw AR h1 NLL
`6.187630`. After that imperfect corrected state was consumed by the next K,
the T--K NLL rose to `11.848419`, `11.768948`, and `11.595691` at horizons
2--4. Therefore the result is not only a first-state readout failure:
recurrent consumption compounds it sharply.

Hard shell projection lowered overall alternating-path NLL by `0.320542` but
reduced raw-AR token agreement to `0.191711` and produced no matching full
block. It is not a remedy for the requested recurrence.

These exponentiated losses are rollout diagnostics, not standard
teacher-forced perplexities. No test data or parameter update was used.
