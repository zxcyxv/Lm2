# Interpretation

The current fast trainer completed 100 fresh optimizer steps with finite loss
and gradients.

| Metric | Result |
|---|---:|
| Trainer wall through final metrics | 50.821 s |
| External process wall | 57.349 s |
| Mean trainer wall per optimizer step | 0.5082 s |
| Byte tokens/s | 34,254 |
| CE labels/s | 32,239 |
| Peak allocated VRAM | 8.003 GB |

Linear extrapolation of the trainer wall gives `50.8 minutes` for 6000 steps
before restoring the normal 64-example periodic evaluations. Including those
evaluations and checkpoint/report overhead, the practical 13M H16 ETA is about
`53--55 minutes` on this GPU.

The completed old-path continuation reported 31,143 byte tokens/s and 9.559 GB,
so the raw record-to-record throughput increase is 10.0% and peak allocation is
16.3% lower. This is not a clean speed comparison because the old run evaluated
64 examples every 500 steps, whereas this ETA run evaluated one example only at
the endpoints. The controlled same-batch forward/backward benchmark remains the
fair compute comparison: 4.76% faster and 17.11% less peak allocation.

Fresh step-100 raw pre-clip gradient norm was 27.97. This short run was
registered for ETA rather than stability, and the value is finite; it should not
be compared with the trained step-6000 gradient norm.

