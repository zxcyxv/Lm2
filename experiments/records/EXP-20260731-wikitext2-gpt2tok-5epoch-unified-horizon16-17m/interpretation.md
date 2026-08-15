# Unified five-epoch horizon audit

All values below use the epoch-5 checkpoint and the same 965 validation suffixes. Open PPL is a fixed-reference rollout score, not ordinary chain-rule corpus PPL.

| model | H1 open PPL | H16 teacher PPL | H16 open PPL | H16 excess NLL | eval wall s |
|---|---:|---:|---:|---:|---:|
| latent | 328.839797 | 322.540679 | 1834.978415 | 1.738559 | 5.201 |
| transformer | 363.353860 | 362.529676 | 7585.047997 | 3.040828 | 37.649 |
| grassmann | 423.857012 | 413.201033 | 12611.733128 | 3.418449 | 52.432 |

## Preregistered criteria

- Latent H1 no worse than both baselines: PASS
- Latent cumulative-H16 excess NLL no worse than both baselines: PASS

The full registered horizon curve remains in `metrics.tsv`; a failed criterion is retained as evidence rather than filtered.
