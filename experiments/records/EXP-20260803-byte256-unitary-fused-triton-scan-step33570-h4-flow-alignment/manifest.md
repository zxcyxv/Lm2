# Step-33570 H1-H4 flow-alignment audit

## Status

- State: completed
- Parent: completed 13M fused-scan ten-epoch run
- Split: fixed validation only; test unread

## Question and comparison

At epoch 10, do H1-H4 innovations align with the causal gold-encoder
trajectory strongly enough to motivate a latent flow matching objective?
Compare step 33570 with step 10071 on identical fixed validation examples and
the identical fused recurrence.  Do not use H5-H16 evidence.

## Fixed protocol

- checkpoints: step 10071 and step 33570
- seed 1337; preserved 64 validation starts; final prefix position 255
- horizons: H1 through H4 only
- fused Triton rotating-frame scan for both checkpoints
- raw target: `g_r - R g_(r-1)`
- ray target: `normalize(g_r) - R normalize(g_(r-1))`
- apply the same `R^-r` coordinate transform to predicted and target velocity
- report raw/ray cosine, relative error, norm, endpoint direction cosine,
  decoded-byte correctness, horizon partitions, and paired sample uncertainty
- metric artifacts are TSV; interpretation is Markdown

## Interpretation gates

- deterministic velocity alignment is a prerequisite, not proof, of full
  distributional flow matching
- require a consistent sign and magnitude across H1-H4, not an aggregate
  produced by one horizon
- endpoint direction alone is insufficient when innovation magnitude is near
  zero
- assess correctness association within horizons to avoid the easier-H1
  mixture confound
