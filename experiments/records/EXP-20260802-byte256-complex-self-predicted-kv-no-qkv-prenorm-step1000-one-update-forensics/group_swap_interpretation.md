# One-update parameter-group swap

This audit restored the native step-1000 AdamW and training-data generator
states, drew the exact next batch, and reproduced update 1001 independently.
The reproduced post-update model matches the main forensic audit with maximum
parameter error `0`. The union of all 18 atomic deltas also has state and
forward-metric error `0`.

The raw pre-clip gradient norm was `69.3633`; clipping used multiplier
`0.01441685`. Same-batch CE changed `3.181236446 -> 3.195414782`.

## Aggregate separation

| Applied post-step delta | CE change | H1 P RMS change | H1 W RMS change | H16 P RMS change | H16 W RMS change |
|---|---:|---:|---:|---:|---:|
| encoder only | +0.007308 | -0.07565 | -44.77 | approximately 0 | approximately 0 |
| central only | +0.004243 | -0.06391 | -27.53 | +0.03172 | +2.283 |
| full update | +0.014178 | -0.12296 | -61.03 | +0.03172 | +2.283 |

Encoder groups held `41.93%` of squared raw gradient but `74.92%` of actual
Adam movement. Central groups held `58.07%` of squared gradient and `25.08%`
of movement. Optimizer history therefore matters as much as the current raw
gradient when attributing this update.

The encoder deltas alter the raw root and H1 but have essentially disappeared
from H16 P/W. Central deltas affect both H1 and the late recurrent regime. This
supports a local separation between root-boundary conditioning and a recurrent
late-state regime; it does not prove a global attractor.

The largest atomic CE worsening was `encoder.block1.attn.qkv` (`+0.006149`).
The largest atomic H1 scale reductions came from `encoder.block1.ffn.2`
(`P -0.03652`, `W -16.45`). Central query and output were major H16 scale-
increase contributors. Phase effects were negligible.

The full update reduced H1 P/W/logit scale while worsening CE. Scale reduction
is therefore not a locally sufficient proxy for likelihood improvement. The
full CE effect is also about `0.00263` larger than encoder-only plus central-
only effects, demonstrating non-additive interaction.

These hybrids isolate the actual one-step Adam deltas on the same batch. They
do not represent alternative optimizer histories, retraining, historical
step 900, or long-run stability.
