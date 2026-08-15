# EXP-20260805 fused-scan horizon curvature and cascade analysis

## Status and evidence scope

- State: completed retrospective/post-hoc analysis
- Producer: `analyze_byte256_parallel_scan_horizon_cascade.py`
- Parent: `EXP-20260803-byte256-unitary-fused-triton-scan-h16-10epoch-13m-rtx5090`
- Seed: 1337
- Split: WikiText-103 byte training trajectory and the parent's fixed
  64-example validation set; test unread

The parent `manifest.md` still says `registered; execution pending`, while
`epoch-checkpoint-run/metrics.tsv` contains the complete epoch-checkpoint
restart through step 33570.  This conflicting status evidence is preserved:
the parent is not edited, and this audit treats only the completed restart TSV
as authoritative numerical evidence.  The interrupted root-level
`metrics.tsv` is deliberately excluded rather than merged with the restart.

This question was not preregistered before the ten-epoch training run.  It is
therefore a descriptive retrospective analysis, not a confirmatory experiment.

## Questions

1. For each H1--H16 validation NLL series, does the epoch-scale decrease
   flatten in the mathematical sense of negative velocity and positive second
   difference?
2. Is there descriptive timing evidence that shallow-horizon improvement
   precedes deeper-horizon improvement, especially H2 preceding H16, or that
   shallow improvement is followed by acceleration of the deeper improvement
   rate?

No intervention changes horizon supervision, and no matched ablation removes
the shallow losses.  Lead--lag statistics therefore cannot establish that a
shallow horizon causes or drives a deeper horizon.

## Fixed input and selection

The producer reads only:

```text
experiments/records/EXP-20260803-byte256-unitary-fused-triton-scan-h16-10epoch-13m-rtx5090/epoch-checkpoint-run/metrics.tsv
```

Its registered SHA-256 is
`e08ec803c181c13e389f974e8beb27206c2d02d2847d34bd2064cb42c5e71025`.
The selected steps are exactly `3357, 6714, 10071, 13428, 16785, 20142,
23499, 26856, 30213, 33570`, representing epochs 1--10.  Irregular early
reports at steps 0, 100, 300, and 1000 are excluded from finite differences.
Every selected `block_nll` must equal the arithmetic mean of H1--H16 within
`1e-12`.

The parent protocol uses width 1344, 13,215,008 parameters, H16 with anchor
stride 16, effective batch 64, a fused Triton rotating-frame scan, 100-step
warmup, and cosine learning-rate decay over 33570 steps.  The training windows
are sampled with replacement, so an epoch here is the parent's fixed token
budget equivalence rather than a without-replacement pass.

## Sign convention and statistics

For horizon `h` and epoch `e`, let its NLL be `L[h,e]`.  On the equally spaced
epoch grid:

```math
v_{h,e}=L_{h,e}-L_{h,e-1},
\qquad I_{h,e}=-v_{h,e},
```

where decreasing NLL has `v < 0` and positive improvement has `I > 0`.  The
discrete NLL acceleration is

```math
a_{h,e}=v_{h,e+1}-v_{h,e}
       =L_{h,e+1}-2L_{h,e}+L_{h,e-1}.
```

Thus a decreasing curve that flattens has `v < 0` and `a > 0`.  The change in
the positive decrease rate is `Delta I = -a`; it is negative when improvement
decelerates.  Per-horizon summaries report raw local signs, means and medians,
the OLS trend of interval velocity, and a Theil--Sen velocity trend.  None of
these signs is a run-success criterion.

Positive-improvement timing reports the maximum-improvement interval and the
centroid of `max(I, 0)` over interval midpoints.  Negative intervals are kept
in all velocity and correlation tables; only this named timing centroid clips
them to zero.

For cascade analysis, lag `+1` correlates `I_source[t]` with
`I_target[t+1]`, so the source leads by one interval.  Lag `-1` is the reverse
direction.  Tables include raw improvement, a full-series linear-detrended
sensitivity, a heuristic normalization by the registered cosine
learning-rate area, and first-interval exclusion.  A separate table correlates
`I_source[t]` with `I_target[t+1]-I_target[t]`; positive correlation would be
descriptively consistent with shallow improvement preceding acceleration of
the target improvement rate.

Because raw `I` is expected to shrink under ordinary diminishing returns, the
retrospective also registers six positive common-envelope variants:

1. signed H1--H16 mean, which is the block-NLL improvement;
2. signed H1--H4 mean;
3. signed H1--H4 median;
4. signed H1--H16 median;
5. average cosine-schedule learning rate divided by peak learning rate;
6. a smooth log-linear exponential fit to block improvement.

For every envelope `g_t`, the signed relative speed is `r_h,t=I_h,t/g_t`.
Its first difference and time slope are called relative acceleration/trend in
this record.  Signed regressions remain negative and are never silently
discarded.  A separate compositional statistic uses

```math
q_{h,t}=\frac{\max(I_{h,t},0)}
              {\sum_j\max(I_{j,t},0)}
```

to ask whether the mass of positive improvement shifts from shallow to deep
horizons.  This positive-only share is explicitly labeled and does not replace
the signed series.

For residual lead--lag, each source and target improvement series is regressed
on `intercept + common envelope`.  The source residual at interval `t` is
paired with the target residual at `t+1`; a partial correlation/regression also
controls the next-interval envelope.  The first transition is separately
excluded as a sensitivity.  This is a small-sample exploratory residualization,
not a Granger-causality test.

All ratio denominators must be finite and strictly positive.  Their minimum,
maximum, and maximum/minimum condition ratio are reported.  A `>=20x` shrinkage
flag marks variants whose late ratios can strongly amplify checkpoint noise;
it is a warning rather than an exclusion.  A power-law envelope is not fit
because nine intervals do not justify another flexible post-hoc family.

The learning-rate normalization is not an optimizer-time causal correction.
There are only nine improvement intervals and eight one-lag pairs before
sensitivity exclusions.  Horizons share weights and validation tokens,
adjacent differences share checkpoints, the learning-rate schedule supplies a
common time trend, and no per-example NLL is available for standard errors.
All correlations are descriptive and are not independent hypothesis tests.

## Outputs

- `run.tsv`: source checksum, selection, seed/split, and fixed constants
- `epoch_nll.tsv`: tidy selected H1--H16 NLL values
- `interval_metrics.tsv`: velocity, positive improvement, and heuristic
  learning-rate normalization for 144 horizon-interval rows
- `curvature_metrics.tsv`: 128 local second differences and the opposite-sign
  improvement-rate changes
- `horizon_summary.tsv`: 16 per-horizon endpoint, curvature, trend, and timing
  summaries
- `cascade_correlations.tsv`: contemporaneous and +/-1 interval correlations,
  transforms, and first-interval sensitivity
- `cascade_acceleration.tsv`: shallow improvement versus subsequent target
  improvement-rate change
- `common_envelopes.tsv`, `envelope_summary.tsv`: all six common decay
  envelopes and denominator stability diagnostics
- `relative_improvement_metrics.tsv`, `relative_acceleration_metrics.tsv`:
  signed and positive-only envelope-relative speeds and their interval changes
- `relative_horizon_summary.tsv`, `relative_horizon_consensus.tsv`: early/late,
  OLS, Theil--Sen, and cross-envelope sign robustness for every horizon
- `horizon_share_metrics.tsv`, `horizon_share_summary.tsv`, and
  `horizon_group_share_summary.tsv`: positive-improvement composition and the
  shallow/deep mass shift
- `residual_lead_lag.tsv`: envelope-residual one-interval lead correlations and
  partial regression sensitivity
- `aggregate_metrics.tsv`: compact cross-horizon and block summaries
- `interpretation.md`: conclusions kept separate from the metric tables

## Result-independent success criteria

- the source checksum, required columns, and all ten exact epoch steps match;
- every selected and derived numeric value is finite;
- output row counts are exactly 160 epoch, 144 interval, 128 curvature, 16
  horizon-summary, 360 raw lead--lag, 80 raw cascade-acceleration, 54 common
  envelope, 6 envelope-summary, 864 relative-improvement, 768 relative-change,
  96 relative-summary, 16 relative-consensus, 144 horizon-share, 16
  horizon-share-summary, 4 group-share, and 240 residual-lead rows;
- block NLL agrees with the H1--H16 mean within the registered tolerance;
- a second producer invocation is byte-identical for every generated output;
- the producer reads no checkpoint or test data and does not edit the parent
  record.

Success does not require positive curvature, a particular correlation sign,
or evidence for the cascade hypothesis.  Conflicting or null timing evidence
is retained with its scope rather than filtered out.

## Reproduction

```bash
python analyze_byte256_parallel_scan_horizon_cascade.py
```
