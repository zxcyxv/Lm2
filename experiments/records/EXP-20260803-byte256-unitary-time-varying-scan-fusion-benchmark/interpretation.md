# Time-varying scan backend result

## Algebra and numerical agreement

The rotating-frame backend evaluates the same time-varying recurrences as the
eager Hillis--Steele implementation:

```text
S_h = U_h S_(h-1) + W_h
z_h = R_h z_(h-1) + delta_h
```

It changes coordinates by the cumulative unitary products, takes an ordinary
causal cumulative sum there, and rotates every prefix back. It therefore keeps
all sixteen literal successor states and does not replace them with direct
horizon predictions.

On the fixed step-1000 checkpoint and batch, eager and rotating-frame CE were
both `3.0406756401`. The maximum logit difference was `8.01e-5` and relative
logit L2 error was `4.42e-7`. Full parameter-gradient cosine was `0.999951`;
the relative gradient L2 difference was `0.00988`. The latter is float32
reassociation amplified by the full inverse-decoder backward, not a different
symbolic recurrence.

The single-kernel Triton rotating-frame implementation produced the same CE
and logit error. Its full-gradient cosine versus eager was `0.999952`, relative
gradient L2 difference `0.00980`, and maximum absolute gradient difference
`1.53e-4`. Small-model tests also compare its memory, latent, root, increment,
and phase gradients directly against the literal recurrence.

## Controlled full-loss timing

The measurement includes encoder, H16 central recurrence, exact inverse
decoder, CE, and backward for batch 64 and stride 16. It excludes sampling,
clipping, and the optimizer step.

| Backend | Median total | Relative to eager |
|---|---:|---:|
| Feedback recurrence | 113.642 ms | 1.546x |
| Eager Hillis--Steele | 175.737 ms | 1.000x |
| Generated associative | 143.868 ms | 1.222x |
| Triton associative | 124.485 ms | 1.412x |
| Rotating-frame cumsum | 126.953 ms | 1.384x |
| Fused Triton rotating frame | 113.845 ms | 1.544x |

The rotating-frame form reduces eager-scan time by `27.8%`, or raises
throughput by `38.5%`. It is still `11.8%` slower than the ordinary feedback
recurrence and `2.0%` slower than the existing Triton associative kernel at
H16. Removing the generic scan's 49 pair compositions is therefore a large
gain, but PyTorch's separate complex phase, multiply, cumsum, and multiply
kernels do not beat the fused Triton implementation at this horizon.

Fusing cumulative phase construction, inverse rotation, inclusive addition
scan, and forward rotation removes those intermediate materializations. The
fused rotating-frame backend is `11.5%` faster than unfused rotating frame,
`9.3%` faster than the generic Triton affine scan, and `54.4%` faster than
eager scan by throughput. Its `113.845 ms` total is only `0.18%` slower than
the `113.642 ms` feedback measurement, which is within practical timing noise.
The forward is one Triton kernel per memory/latent carrier. Each corresponding
backward kernel fuses the reverse frame suffix scan, increment/root adjoints,
and shared phase-gradient reduction.

## Direct trainer timing

The 200-step ordinary trainer confirmation included sampling, gradient
clipping, fused AdamW, and startup overhead. Its cumulative time was
`26.6226 s`, or `0.13311 s/step`. After the first 100 steps, the second
100-step segment took `0.12772 s/step`, compared with `0.17687 s/step` for
the preserved eager scan's step-300 to step-1000 segment. This independently
reproduces a `1.385x` steady-state speedup.

At the measured steady rate, 6000 optimizer steps take about `12.8 minutes`
excluding validation, versus `17.7 minutes` for eager scan: approximately
`4.9 minutes` saved. The ordinary state-dependent feedback trainer remains
faster at about `11.6 minutes` per 6000 steps.

The short rotating-frame run reached train CE `3.2505` and validation block
NLL `3.2235` at step 200. Its early float32 training trajectory need not be
pointwise identical to the eager run, so this timing confirmation is not used
as a substitute for the preserved 1000-step scan quality record.

The fused backend's direct 200-step run took `23.8419 s` cumulatively, or
`0.11921 s/step` including startup. Its second 100-step segment took
`0.11452 s/step`, corresponding to about `11.45 minutes` for 6000 steps and
`152006` nominal sampled bytes/s. This is a `1.544x` steady speedup over eager
scan and a `1.115x` speedup over unfused rotating frame. At step 200 it had
finite train CE `3.2561`, gradient norm `5.7929`, and validation block NLL
`3.2534`.
