# Interpretation

## Evidence boundary

This is the exact next optimizer update from the native step-1000 checkpoint.
It uses the checkpointed data-generator and fused-AdamW states. It is not a
reconstruction of the historical step-900 spike, and frozen-weight variants
do not establish what a newly trained trajectory would do.

All full-update gradients below are for a CE already averaged over batch 64,
16 anchors, and 16 horizons. The reported norm is therefore not an unnormalised
sum over 16,384 labels.

## Exact update 1001

- Same-batch CE before update: `3.181236461`.
- Raw global gradient norm: `69.3632843`.
- Clip multiplier for norm 1: `0.01441685`.
- Same-batch CE after the clipped AdamW update: `3.195414748`.

The update made its own training batch worse by `0.0141783`. The sign of that
finite loss change alone does not tell whether the actual delta was locally an
ascent direction or crossed curvature after initially descending.

The preregistered follow-up in `update_direction_interpretation.md` narrows
that statement: the actual delta was not first-order ascent. It had
`g dot delta = -7.15459` and cosine(`g`, `delta`) `-0.505824`, so it remained
a first-order descent direction despite the finite same-batch CE increase.
Clipping itself is a positive scalar rescaling; the observed sign reversal is
therefore a finite-step Taylor-remainder effect at this point, not a reversal
of the current-gradient directional derivative.

The no-training follow-up in `update_line_search_interpretation.md` evaluates
the same batch along scaled copies of that exact Adam delta. Its lowest
registered float32 point is near alpha `0.0068`; the robust coarse baseline
sign change lies between alpha `0.075` and `0.1`, while alpha `1` is worse.
The tiny-alpha curve is deterministic but quantization-jagged, so neither the
sampled minimum nor the narrower bisection sign change is treated as a smooth
stationary-point/root estimate.

The raw gradient was concentrated in six groups:

| Group | Gradient L2 | Squared global share |
|---|---:|---:|
| central query | 31.4756 | 20.59% |
| RevBlock 0 FFN | 30.9465 | 19.91% |
| central key | 30.0108 | 18.72% |
| RevBlock 1 attention QKV | 26.2411 | 14.31% |
| central output | 23.0317 | 11.03% |
| central value | 19.2614 | 7.71% |

These shares sum with the remaining groups to exactly one. The largest actual
Adam movement was instead RevBlock 1 FFN (`delta L2 = 0.115843`), showing that
the stored optimizer state materially changes the ordering between current
gradient size and parameter movement.

## Where the large gradient came from

The `1/16`-weighted H5 and H6 gradients had L2 norms `21.0585` and `36.9237`.
Their cosine was `0.997262`, so they reinforced rather than cancelled. Across
all 16 horizons, the coherent-sum/RSS ratio was `1.57463`; 21 of 120 distinct
horizon pairs were opposed, but the dominant pair was strongly aligned. The
sum of the separately computed horizon gradients reproduced the total with
relative error `1.81e-4`.

Thus this batch's norm was not a uniform sixteen-step inflation. It was a
localized, coherent H5/H6 parameter-sensitivity event. The first-microbatch
activation trace does not identify which of the remaining physical rows caused
that full-batch event; no per-example attribution is claimed here.

## Forward scale and recurrent Jacobians are different questions

The final stored carriers stayed at RMS approximately one because the recurrent
boundary is normalized. Raw intermediate scales were nevertheless highly
nonuniform:

| Horizon | P RMS | innovation W RMS | raw innovation delta RMS | logit RMS |
|---:|---:|---:|---:|---:|
| H1 | 1.9428 | 398.075 | 3633.32 | 15.8346 |
| H2 | 1.0348 | 80.8878 | 304.776 | 8.6298 |
| H3 | 0.6421 | 36.8054 | 199.436 | 7.1544 |
| H4 | 0.5767 | 21.2257 | 7.5808 | 6.9094 |
| H16 | 0.5753 | 20.9190 | 7.1468 | 6.5752 |

So the boundary norm prevents stored-state divergence, but it does not bound
P, W, or the raw pre-normalization update. The trajectory enters a much smaller
scale regime around H4.

The matrix-free registered probe, on one fixed example/anchor and in the
audit's hidden/memory-balanced product norm, found local one-step singular
values growing from `1.60` at H1 to approximately `10.08` at H10--H16. In
contrast, that root's composed gain shrank monotonically from `1.60` at H1 to
`6.51e-6` at H16 (`3.92e-6` after the update). At this probed trajectory, the
locally expanding singular directions therefore do not align with the tangent
propagated from the root.

This probe does not support a simple claim that every root-state Jacobian
uniformly explodes through sixteen compositions, but one root cannot rule it
out batch-wide. It also does not rule out large tied-parameter gradients: the
same Q/K/V/O and encoder/decoder weights are injected at every occurrence, and
their use-site gradients add. H5/H6 show that this addition can be coherent
even while the probed root-state tangent contracts.

## Frozen conditioning checks

| Condition | CE | Raw gradient L2 | Ratio to attached raw |
|---|---:|---:|---:|
| attached, raw | 3.1812365 | 69.3633 | 1.000 |
| horizon-detached, raw | 3.1812364 | 20.1422 | 0.290 |
| attached, sqrt(write-count) | 3.1854992 | 12.7377 | 0.184 |
| detached, sqrt(write-count) | 3.1854992 | 12.7491 | 0.184 |

Detach leaves this forward pass unchanged and removes cross-horizon gradient
paths. Sqrt scaling changes both the forward function and objective, so its
gradient norm is not a pure rescaling of the historical gradient. On these
frozen weights it dominates the conditioning change, because adding detach on
top changes `12.7377` only to `12.7491`. None of these rows says whether either
variant learns better or prevents a future spike after retraining.

## What the actual parameter update changed

The independent parameter-group swap exactly reproduced the post-update model.
Encoder groups held `41.93%` of squared raw gradient but `74.92%` of Adam
movement. Central groups held `58.07%` of squared raw gradient and `25.08%` of
movement.

- Encoder-only deltas changed CE by `+0.007308`, H1 P by `-0.07565`, and H1 W
  by `-44.77`; their H16 P/W effects were approximately zero.
- Central-only deltas changed CE by `+0.004243`, H1 P by `-0.06391`, and H1 W
  by `-27.53`; they changed H16 P by `+0.03172` and H16 W by `+2.283`.
- The full update changed H1 P `1.9428 -> 1.8199`, W `398.1 -> 337.0`, and
  logit RMS `15.83 -> 15.06`, while CE worsened.
- The largest isolated CE worsening was RevBlock 1 attention QKV:
  `+0.006149`.
- Phase-only deltas were negligible at this update.

The encoder delta therefore acts primarily through the encoded-root/H1
boundary, while the late recurrent regime is primarily sensitive to central
parameters. The full CE change exceeds the encoder-only plus central-only
changes by about `0.00263`, so the simultaneous update has material nonlinear
interaction. Raw-scale reduction is not locally monotone with likelihood.

## Validation and scope

- Parameter-group squared gradients reproduce the measured global norm.
- Horizon-gradient sum relative error: `1.81e-4`.
- Shared RevNet Linear call-site reconstruction maximum relative error on the
  traced microbatch: `2.48e-5`.
- The independently reproduced post-update model and the union of all 18
  atomic group deltas both match with maximum state/metric error `0`.
- Detailed tensor adjoints and encoder/inverse-decoder call roles cover the
  first fixed microbatch of 16 rows; full parameter and horizon gradients use
  all 64 rows.

The strongest supported conclusion is narrow: at native step 1000, stored
state normalization works in the forward carrier, but tied recurrent parameter
sensitivity remains poorly conditioned and can become coherently large at
specific horizons. This one update does not establish which training-time
normalization or detachment policy is best.
