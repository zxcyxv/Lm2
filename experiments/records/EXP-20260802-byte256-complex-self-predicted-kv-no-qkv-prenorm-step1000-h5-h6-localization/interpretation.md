# Interpretation

## Evidence boundary

This is a frozen-weight localization of the native step-1000 exact-next batch. It does not reconstruct step 900, estimate event frequency, or assign a causal mechanism. Every gradient below already contains the historical `1/16` horizon weight and its exact global row/anchor weight.

The first row split changed the physical batch shape from 16 to 1 and failed
vector reconstruction (`1.58%` H5, `6.87%` H6). It is retained in
`failed_shape_attempt.tsv` but excluded from attribution. The accepted split
kept the same 16-row forward and selected one row or anchor only at the loss.

## Full-batch reproduction

H5 L2 was `21.058456` and H6 L2 was `36.9237277`; their cosine was `0.997261948`. Relative disagreement with the parent TSV was `0` for H5, `0` for H6, and `0` for their cosine.

## Physical microbatches

- micro 0: combined L2 `26.1548927`, mean H5/H6 CE `3.37284207`
- micro 1: combined L2 `0.832549998`, mean H5/H6 CE `3.12247527`
- micro 2: combined L2 `1.0458781`, mean H5/H6 CE `3.26069069`
- micro 3: combined L2 `31.7630312`, mean H5/H6 CE `3.08304226`

The dominant physical microbatch was `3`. The four microbatch vectors reconstructed full H5/H6 with relative errors `0` / `0`.

Microbatches 0 and 3 projected onto `45.10%` and `54.80%` of the full H5+H6
gradient and had cosines `0.99921` and `0.99975` with it. Microbatches 1 and 2
had combined norms only `0.833` and `1.046` and near-zero directional cosine.
Thus the event was not spread uniformly across the four physical batches.

## Rows within the dominant microbatch

- row 55: combined L2 `24.4914066`, mean CE `3.07435846`, cosine to full H5+H6 `0.999126836`
- row 58: combined L2 `7.26304022`, mean CE `2.66118288`, cosine to full H5+H6 `0.999056441`
- row 56: combined L2 `0.273070751`, mean CE `2.84745538`, cosine to full H5+H6 `0.003419195`
- row 59: combined L2 `0.261389063`, mean CE `3.44478893`, cosine to full H5+H6 `0.0138888027`
- row 53: combined L2 `0.252279171`, mean CE `3.02045298`, cosine to full H5+H6 `0.00369121562`

The dominant row was `55`. The 16 row vectors reconstructed their microbatch with relative errors `5.17e-05` / `4.95e-05`. A large row norm alone is not evidence that its loss or any one raw forward scale caused the full event; the TSV retains their separate measurements and directions.

Rows 55 and 58 supplied almost all of microbatch 3's aligned component. Their
projection coefficients onto the full H5+H6 vector were `0.42229` and
`0.12522`; the other 14 rows had individual coefficients below `1e-4` in
magnitude. Their mean CE values (`3.074` and `2.661`) were not extreme, so loss
magnitude alone does not explain the parameter sensitivity.

## Anchors and forward scales

The largest selected-row anchor was ordinal `0` (anchor position `0`), with combined gradient L2 `24.4905185`. Anchor reconstruction relative errors were `1.59e-05` / `1.41e-05`. `row_forward_metrics.tsv` contains P, W, Sraw, Zraw, stored carriers, decoder and logit scales for every row in the selected microbatch. `anchor_metadata.tsv` retains token IDs, losses, confidence, and the same scales per anchor. These are co-located observations, not a causal rank.

Anchor 0 alone had virtually the whole row-55 norm (`24.4905` versus
`24.4914`) and projected onto `42.22%` of the full vector. At this anchor,
H5 `Zraw`/`Sraw` denominators were `0.39286`/`1.21411`, versus medians
`7.38585`/`21.7222` over the other 15 anchors. One step later its H6
`P/W/Zraw/Sraw` RMS values were `1.17061/78.7418/73.0970/79.7123`; the
corresponding other-anchor medians were `0.57313/20.7831/7.41329/21.7818`.
Stored Z and S still had RMS one. These scale changes are exactly co-located
with the gradient event, but this audit does not identify which derivative or
parameter path made them causal.

Rows 55 and 58 both start with byte ID 104 (`h`) at anchor 0, so their causal
anchor-0 forward tensors are exactly identical while their H5/H6 target bytes
differ. This separates the shared forward trajectory from the target-specific
CE adjoint; it does not yet localize the large secondary microbatch-0 component.
