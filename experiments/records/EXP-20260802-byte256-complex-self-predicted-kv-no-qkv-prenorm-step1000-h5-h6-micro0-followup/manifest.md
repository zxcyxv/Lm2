# EXP-20260802 step-1000 H5/H6 microbatch-0 follow-up

## Status

- State: completed; native microbatch-0 follow-up executed
- Primary localization:
  `../EXP-20260802-byte256-complex-self-predicted-kv-no-qkv-prenorm-step1000-h5-h6-localization/`
- Parent one-update audit:
  `../EXP-20260802-byte256-complex-self-predicted-kv-no-qkv-prenorm-step1000-one-update-forensics/`
- Source checkpoint: ignored native `step1000.pt`
- Test split remains unmaterialized and unread.

## Question

The primary localization found that physical microbatch 0 supplied combined
H5+H6 gradient L2 `26.1549`, cosine `0.99921` with the full vector, and
projection coefficient `0.45101`, second only to microbatch 3. Does this
secondary component also localize to one or a few rows/anchors with the same
observed pattern as microbatch 3: an unusually small H5 hidden-boundary
denominator followed by an H6 P/W/Sraw/Zraw scale jump?

The row, anchor, input byte, target adjoint, and mechanism are not selected in
advance. Pattern agreement is descriptive; it is not preregistered as proof
that boundary normalization causes the gradient.

## Fixed protocol

- Reuse the primary audit's native checkpoint loader, exact-next batch,
  H5/H6 gradient weighting, vector summaries, and forward instrumentation.
- Select original physical microbatch 0: global rows 0--15.
- Preserve the original batch-16 forward shape for every differentiation and
  select only a row/anchor loss element. Never rerun a row as batch shape 1.
- Preserve attached cross-horizon gradients, raw accumulated reads, strict
  float32, disabled TF32, and no optimizer update.
- Every gradient contribution retains weight
  `1 / (64 rows * 16 anchors * 16 horizons)` per label.
- Split all 16 rows, choose the largest combined-H5+H6 row, then split all 16
  anchors in that row.
- Record the same H5/H6 per-row and per-anchor P, W, Sraw, Zraw, stored-carrier,
  decoder, logit, and boundary-denominator scales as the primary audit.

Train SHA-256:
`062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`.
Seed and exact data-generator state come from the checkpoint.

## Comparisons and success criteria

The fixed reference is the primary audit's microbatch-0 H5 and H6 parameter
vectors. The follow-up succeeds only if:

- the 16 row vectors reconstruct microbatch-0 H5 and H6 with relative L2 error
  below `5e-4`;
- the selected row's 16 anchor vectors reconstruct its H5 and H6 gradients
  below `5e-4`;
- the dominant row/anchor gradient L2, cosine and projection against full H5,
  H6 and H5+H6 are retained;
- H5 and H6 forward scales are reported for every row and anchor in this
  microbatch, including conflicting or non-finite values.

Pattern match against microbatch 3 is reported using direct ratios: selected
anchor H5 Zraw/Sraw denominators relative to the other-anchor medians, and H6
P/W/Zraw/Sraw relative to those medians. No threshold is used to relabel a
mixed result as success or failure.

## Outputs

- `scope_gradients.tsv`
- `gradient_alignments.tsv`
- `row_forward_metrics.tsv`
- `anchor_metadata.tsv`
- `pattern_comparison.tsv`
- `interpretation.md`

Metrics remain TSV and interpretation remains Markdown. Checkpoints and tensor
artifacts remain ignored and are not added to Git. Existing records are not
overwritten.

## Evidence boundary

This is a post-primary, preregistered follow-up on one already-identified
physical microbatch. It tests repeatability within the same native batch, not
across batches or checkpoints, and does not estimate failure frequency.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python eval_byte256_complex_self_predicted_kv_no_qkv_prenorm_step1000_h5_h6_micro0_followup.py
```
