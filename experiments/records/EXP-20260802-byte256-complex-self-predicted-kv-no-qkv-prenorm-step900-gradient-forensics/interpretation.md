# Interpretation

The missing historical step-900 checkpoint could not be reconstructed exactly
from the preserved step-500 payload. Consequently, the requested internal
gradient attribution has not been run and no failed-replay snapshot is treated
as evidence about the original step-900 event.

## Verification failure

| Step | Source CE | Replay CE | Source gradient norm | Replay gradient norm |
|---:|---:|---:|---:|---:|
| 600 | 3.100222 | 3.078346 | 26.386745 | 39.878647 |
| 700 | 3.078413 | 3.153504 | 10.648123 | 33.773705 |
| 800 | 3.065093 | 3.074492 | 3.379766 | 6.306804 |
| 900 | 10.958057 | 3.063244 | 1563.952271 | 2.327550 |

All eight comparisons fail the registered `1e-5` relative or `1e-6`
absolute tolerance. The replay did not reproduce the source event: its maximum
gradient norm was `121.563690` at step 572, and after step 800 its maximum was
only `22.286274` at step 834.

## What remains valid

The preserved source TSV still establishes that the original training process
had a pre-update-900 gradient norm of `1563.952271`, followed by the recorded
post-update validation failure. It does not contain the internal activations or
the model weights at that instant. The new `pre_update0900.pt` and
`post_update0900.pt` contain a different, non-exploding replay trajectory and
must not be substituted for those missing historical weights.

The restart restored the model, fused AdamW state, and dedicated sampling
generator. It could not restore information absent from the checkpoint:
global CPU/CUDA RNG states, deterministic CUDA/SDPA settings and backend state,
or an immutable checksum of the exact producer and common source. CUDA
scaled-dot-product attention was used without deterministic-algorithm/backend
pinning. Any of these small restart differences can be amplified by this
highly sensitive recurrence; this audit does not identify which one caused the
trajectory split.

An exact forensic decomposition now requires either a native checkpoint from
immediately before update 900 or a fresh, separately registered run that saves
complete restart state and dense checkpoints around every instability
threshold.
