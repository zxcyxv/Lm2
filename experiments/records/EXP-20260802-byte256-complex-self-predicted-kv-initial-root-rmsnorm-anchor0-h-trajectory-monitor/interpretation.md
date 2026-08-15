# Interpretation

## Fixed validation sample

The registered seed selected 256 validation windows; 6 (0.023438) have byte 104 at anchor 0. The immutable sample was passed to candidate and control in the same batch-16 grouping.

Sample token SHA-256: `459b033a1100069cf276f885eb70bab39cca4990dfcf16a88281822c269dd91b`.

## Registered H5/H6 view

Ratios are descriptive. No ratio threshold was preregistered as a success criterion.

| Step | Tensor | Candidate h | Control h | Cand/control | Cand h/non-h | Cand h/other anchors | Control h/non-h | Control h/other anchors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | H5 Z boundary denominator | 0.0148695 | 0.0545526 | 0.272572 | 0.650842 | 0.59316 | 0.569075 | 0.473823 |
| 100 | H5 S boundary denominator | 1.00003 | 1.02209 | 0.978412 | 0.997554 | 0.998014 | 0.927279 | 0.894252 |
| 100 | H6 P RMS | 0.125564 | 0.22871 | 0.54901 | 0.880886 | 0.787474 | 0.834454 | 0.769754 |
| 100 | H6 W RMS | 0.0155715 | 0.0779556 | 0.199749 | 0.811678 | 0.650839 | 0.681157 | 0.594518 |
| 100 | H6 Zraw RMS | 0.0130201 | 0.0524163 | 0.248397 | 0.607072 | 0.585914 | 0.652275 | 0.590724 |
| 100 | H6 Sraw RMS | 1.00056 | 1.02676 | 0.974491 | 0.998277 | 0.997323 | 0.944627 | 0.920969 |
| 500 | H5 Z boundary denominator | 0.369884 | 1.77535 | 0.208344 | 0.927806 | 0.920573 | 1.0168 | 1.01019 |
| 500 | H5 S boundary denominator | 2.8498 | 7.48544 | 0.380713 | 0.932431 | 0.924957 | 1.01256 | 1.00739 |
| 500 | H6 P RMS | 0.341221 | 0.4829 | 0.706608 | 0.928158 | 0.933427 | 0.998246 | 0.999508 |
| 500 | H6 W RMS | 1.82441 | 6.35446 | 0.287107 | 0.829878 | 0.843621 | 0.996475 | 0.998926 |
| 500 | H6 Zraw RMS | 0.381678 | 1.72903 | 0.220747 | 0.877937 | 0.879228 | 0.993535 | 0.996722 |
| 500 | H6 Sraw RMS | 2.82386 | 7.35338 | 0.384022 | 0.883856 | 0.893322 | 0.996955 | 0.999074 |
| 1000 | H5 Z boundary denominator | 0.0730073 | 0.392512 | 0.186 | 0.978139 | 0.96221 | 0.053147 | 0.0529449 |
| 1000 | H5 S boundary denominator | 1.51837 | 1.2134 | 1.25133 | 1.01787 | 1.00398 | 0.0558476 | 0.0557109 |
| 1000 | H6 P RMS | 0.0804502 | 1.17052 | 0.0687304 | 0.973653 | 0.97243 | 2.04252 | 2.04072 |
| 1000 | H6 W RMS | 0.397744 | 78.7301 | 0.00505199 | 0.955275 | 0.953388 | 3.78857 | 3.78284 |
| 1000 | H6 Zraw RMS | 0.0606074 | 73.0833 | 0.000829292 | 0.975666 | 0.96682 | 9.85874 | 9.83604 |
| 1000 | H6 Sraw RMS | 1.39747 | 79.7006 | 0.017534 | 0.987071 | 0.986525 | 3.6594 | 3.65411 |

## Result

At step 1000, candidate/control anchor-0 byte-h H6 innovation-W and raw-successor-Z ratios were 0.00505199 and 0.000829292.

The control byte-h H6 event was absent from the candidate: its byte-h W/Zraw means were near its own non-h rows, and the largest candidate byte-h/non-h mean ratio across H2--H16 P/W/Zraw/Sraw was 1.05261. The candidate also shifted global trajectory scales downward, so this supports removal rather than relocation of the registered event without establishing a unique causal pathway.

## Completion checks

- Every requested exact candidate/control checkpoint pair loaded strictly; no `last.pt` or nearest-step substitution was used.
- All four registered groups were non-empty and all H0/H1--H16 trajectory summaries were finite.
- Raw and stored successor Z/S scales remain separate in `trajectory_metrics.tsv`, including every conflicting horizon.
- `checkpoint_comparisons.tsv` retains candidate/control and within-model h/non-h and h/other-anchor differences and ratios.

## Evidence boundary

This is a fixed validation-forward monitoring audit. It does not establish that initial-root scale caused a historical training spike, and it does not replace likelihood, gradient-norm, or fresh-seed training comparisons. Train and test splits were not sampled.
