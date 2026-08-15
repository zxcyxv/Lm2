# Interpretation

## Near-equivalence result

Step 500 rejects the registered near-equivalence claim.

- same-prefix one-block agreement: `0.4141`;
- h1--h4 agreement: `1.0`, `0.3438`, `0.1562`, `0.1562`;
- exact four-token continuation agreement: `0.0156`; and
- 64-token free-running agreement: `0.1245`.

The exact h1 match again verifies that both implementations begin with the
same `T_i=Q_iK`. Their disagreement starts only after AR quantizes the first
state to a token and re-encodes the realized prefix.

## Why this is not a simple regression from step 100

All local agreement numbers after h1 are below step 100, but the step-100
policies had collapsed-sample fraction `1.0` and repeated one token for
almost the entire continuation. Agreement was therefore inflated by a common
token fixed point.

At step 500, output diversity has emerged and exposes the path difference:

| metric | same-operator AR | same-operator block-4 |
|---|---:|---:|
| reference accuracy | 0.0193 | 0.0337 |
| distinct-2 | 0.2904 | 0.3311 |
| immediate repeat | 0.4397 | 0.2458 |
| period-4 repeat | 0.4839 | 0.4542 |
| collapsed fraction | 0.6875 | 0.1406 |
| longest identical run | 19.88 | 3.22 |

Block-4 is currently both more accurate and much less collapsed. This does
not establish good language quality at 500 updates, but it rules out the
claim that the two inference policies are already functionally equivalent.

## Latent metric caveat

The reported state error is not raw MSE. It is computed per state as

`||prediction-target||^2 / ||target||^2`.

This controls target scale and makes horizon comparisons more meaningful.
It is still not an absolute decoder-space metric: jointly learned encoder
coordinates can differ, and equal relative errors can cross very different
LM-head margins depending on direction. The generation audit is therefore
stronger functional evidence than comparing relative latent MSE alone.

## Structural interpretation

Promoting the residual to a reusable operator removes the algebraic
first-step/later-step mismatch:

`u_(j+1) = T_i u_j`

at every open-loop horizon. It does not make the following diagram commute:

`T_i u -> token argmax -> re-encode`

versus

`T_i u -> retain continuous latent state`.

The hard token bottleneck discards continuous information. Moreover, the
current loss directly trains only the open-loop `T_i^j hA` tape, whereas the
re-anchored AR path is absent from training. The observed block advantage is
therefore consistent with the objective rather than an inference-code
accident.

This is evidence against the assumption that a homogeneous branch operator
alone is sufficient for AR equivalence. It does not yet determine whether
the gap closes by step 1000, and it does not justify changing the active
preregistered run.
