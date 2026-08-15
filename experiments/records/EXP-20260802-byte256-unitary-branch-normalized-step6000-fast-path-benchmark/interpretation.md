# Interpretation

The CE-only fast path produced a modest compute improvement and a material
memory reduction on the current GPU at microbatch 64.

| Path | Median forward+backward | Byte tokens/s | Peak allocated VRAM |
|---|---:|---:|---:|
| Full diagnostic | 509.567 ms | 34,162 | 9.521 GB |
| CE-only fast | 486.430 ms | 35,787 | 7.892 GB |

The median speedup was `1.0476x` (4.76%) and peak allocation fell by 17.11%.
The optimization therefore mainly creates batch/width headroom; it does not by
itself produce a large ETA reduction.

The first preregistered default-close loss check failed before metrics were
written: full `2.8909676075` versus fast `2.8908076286`, absolute difference
`1.5998e-4`. The rerun used the explicitly amended finite-precision tolerance
in the manifest and preserves both values in `metrics.tsv`. Prefix-only and
full-length CUDA causal encodes have different matrix shapes, so this result is
consistent with numerical path variation, but it is not recorded as bitwise or
default-tolerance equivalence.

The timing excludes data sampling, clipping, optimizer update, reporting and
validation. It is a compute-path ratio, not end-to-end throughput.

