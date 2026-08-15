# Stochastic value-write continuation interpretation

## Outcome

The low-rank process-noise equation was numerically stable and retained the
fused scan.  A weight-0.01 H1--H4 ray-velocity auxiliary moved every shallow
horizon in the registered target direction with negligible NLL cost, but it
did not make the random tape a used future-mode variable.  Both stochastic
arms slightly reduced rather than increased output sensitivity to noise.

All three arms completed 300 optimizer steps without a non-finite value.  Raw
pre-clip final gradient norms were `0.5854`, `0.5835`, and `0.5841` for
deterministic CE, stochastic CE, and stochastic ray-flow respectively.

## Corrected fixed-validation comparison

The table below uses the identical-full-encode corrected audit, not the
confounded parent noise-delta columns.

| Checkpoint | Deterministic block NLL | Stochastic block NLL | H1--H4 ray relative loss | H1--H4 ray cosine | Noise logit RMS | Noise top-1 disagreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parent step 33570, scale 0.05 | 2.835117 | 2.835173 | 0.997752 | 0.079853 | 0.051129 | 0.001892 |
| Deterministic CE, step 300 | 2.836921 | 2.836921 | 0.997798 | 0.078696 | 0 | 0 |
| Stochastic CE, step 300 | 2.837483 | 2.837567 | 0.997796 | 0.078722 | 0.049792 | 0.001526 |
| Stochastic ray-flow, step 300 | 2.837562 | 2.837643 | 0.997382 | 0.086485 | 0.049362 | 0.001587 |

The stochastic-ray-flow arm's deterministic NLL regressed by `0.002445` from
the parent, well inside the preregistered `0.05` limit.  Against the matched
stochastic-CE arm, its ray relative loss improved by `0.000414`, cosine
increased by `0.007763`, and deterministic NLL differed by only `0.000079`.
The absolute ray-loss change is small because the predicted velocity remains
far below the observed sample-path velocity magnitude.

The directional change was not an aggregate produced by one horizon:

| Horizon | Parent cosine | Stochastic CE | Stochastic ray-flow |
| ---: | ---: | ---: | ---: |
| H1 | 0.041684 | 0.041997 | 0.049958 |
| H2 | 0.028898 | 0.027729 | 0.035568 |
| H3 | 0.097782 | 0.096173 | 0.104135 |
| H4 | 0.151050 | 0.148992 | 0.156277 |

## Interpretation boundary

The experiment establishes three narrow facts.

1. Sampling the complete complex noise tape in advance and adding it through
   `k(v + epsilon)^dagger` preserves the literal recurrence, rotating-frame
   scan, and stable backward path at this scale.
2. Detached ray-velocity supervision can change the learned innovation
   direction without forcing the decoder-visible state to equal a future
   posterior code and without materially damaging CE.
3. The present loss does not assign different noise paths to different future
   modes.  Noise logit RMS fell from `0.051129` to `0.049792` under stochastic
   CE and to `0.049362` under stochastic ray-flow.  Thus the optimizer mostly
   treats the random write as a nuisance to be made robust against.

Consequently this is positive evidence for the stochastic state equation and
for a small velocity auxiliary, but not evidence for distributional flow
matching.  A distributional follow-up would need an explicit noise-to-future
sample coupling or bridge objective; merely injecting independent process
noise and pointing every realization toward the same observed path is
insufficient.

## Artifacts

- `metrics.tsv`: preserved original three-arm reports, including the
  subsequently scoped noise-delta columns
- `run.tsv`: fixed seeds, scales, checkpoint, and optimizer protocol
- corrected audit: sibling record
  `EXP-20260803-byte256-unitary-stochastic-value-ray-flow-corrected-noise-audit`
