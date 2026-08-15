# Latent epoch-5 feedback ablation

This is a post-hoc, within-checkpoint diagnostic. Both paths use the same model and literal prefix; only the feedback representation after the first prediction differs.

- First-step maximum absolute logit difference: `7.62939453e-06`
- First-step top-1 agreement: `1.000000000`

| path | H1 open PPL | H16 open PPL | H16 excess NLL |
|---|---:|---:|---:|
| continuous_state | 328.839804 | 1834.978403 | 1.738559 |
| discrete_argmax_reencode | 328.839802 | 8617.285308 | 3.285296 |

## Registered criteria

- Matched first step: PASS
- Lower continuous-state H16 excess NLL: PASS

These are fixed-reference rollout scores. Increasing horizon withholds gold-token correction for longer, so absolute PPL growth by itself is not evidence for either feedback path.
