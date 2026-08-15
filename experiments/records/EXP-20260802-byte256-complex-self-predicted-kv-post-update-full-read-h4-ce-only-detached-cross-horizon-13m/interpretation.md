# Partial interpretation

The exact-detach run was stopped by user after its step-100 report.

| Metric at step 100 | Detached | Attached |
|---|---:|---:|
| Block validation NLL | 3.117875 | 2.971892 |
| H1 validation NLL | 2.659155 | 2.489069 |
| H2 validation NLL | 3.107959 | 2.973818 |
| H3 validation NLL | 3.310829 | 3.191931 |
| H4 validation NLL | 3.393556 | 3.232749 |
| Peak allocated VRAM, bytes | 7689229312 | 7507823104 |

Forward values passed the registered identity tolerance and synthetic
cross-horizon activation gradients were zero. Exact detachment also made an
H4-only loss provide zero K/V gradient because each innovation is written
after the same horizon's causal P read and affects later horizons only. The
full mean loss retained K/V gradient through the H1 initialization write.

No step-1000 conclusion is claimed.
