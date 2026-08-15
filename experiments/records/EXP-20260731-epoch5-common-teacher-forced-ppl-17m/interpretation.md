# Common epoch-5 teacher-forced PPL

Each model scores the same 246,075 validation next-token targets exactly once with literal gold context. No open rollout or argmax enters this metric.

| model | NLL | PPL | accuracy | tokens | wall s |
|---|---:|---:|---:|---:|---:|
| latent | 5.788582203 | 326.549715 | 0.216430 | 246075 | 4.072 |
| transformer | 5.884523202 | 359.431351 | 0.207079 | 246075 | 3.209 |
| grassmann | 6.022627807 | 412.661567 | 0.193933 | 246075 | 4.046 |

## Registered criterion

- Latent PPL lower than Transformer: PASS
