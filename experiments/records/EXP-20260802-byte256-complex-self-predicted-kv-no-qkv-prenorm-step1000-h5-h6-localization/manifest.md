# EXP-20260802 step-1000 H5/H6 localization

## Status

- State: completed; native step-1000 H5/H6 localization executed
- Parent audit:
  `../EXP-20260802-byte256-complex-self-predicted-kv-no-qkv-prenorm-step1000-one-update-forensics/`
- Source run:
  `../EXP-20260802-byte256-complex-self-predicted-kv-urm-boundary-postnorm-no-qkv-prenorm-h16-ce-only-attached-stride16-13m/`
- Source checkpoint: ignored native `step1000.pt`
- Test split remains unmaterialized and unread.

### Preserved execution amendment

The first row-localization implementation reran each selected row with batch
shape 1. Although its loss weights were correct, those row vectors failed to
reconstruct the original batch-16 physical-microbatch gradient: relative error
was `0.0158288` for H5 and `0.0686923` for H6. Changing the batch shape changes
the SDPA/kernel numerical path, which is inadmissible for attribution of this
high-sensitivity event. Those values are retained in `failed_shape_attempt.tsv`
and were not interpreted.

The accepted execution keeps the original 16-row physical forward shape for
every row and anchor and selects only the loss element before differentiation.
It passed the preregistered reconstruction criteria.

## Question

The parent audit found unusually large, strongly aligned `1/16`-weighted H5
and H6 parameter gradients on the native step-1000 exact-next batch. Is that
event spread across the 64 rows and 16 anchors, or localized to particular
physical microbatches, rows, or anchors?

No parameter group, example, anchor, token, forward scale, or causal mechanism
is preregistered as the explanation.

## Fixed protocol

- Load the same native `step1000.pt` model and restore the checkpointed
  training-data generator.
- Draw exactly the next 64-row window used by the parent one-update audit:
  context 256, window 272, anchor stride 16, H16.
- Strict float32 with TF32 disabled; no optimizer update.
- Preserve attached cross-horizon gradients and the source model's raw
  accumulated-read rule.
- Treat every H5/H6 contribution as it enters the historical total objective:
  horizon weight `1/16` and batch weight `1/64`.
- First split the 64 rows into the four physical 16-row microbatches used by
  training. Then split every row within the dominant physical microbatch.
- If tractable, split the 16 anchors within the dominant row. Dominance is
  selected only by the measured combined H5+H6 parameter-gradient L2.

Train SHA-256:
`062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`.
Seed and exact data-generator state come from the checkpoint.

## Measurements

For each scope and each of H5, H6, and their sum, record:

- weighted parameter-gradient L2 and maximum absolute element;
- dot product and cosine with the full-batch H5, H6, and H5+H6 gradients;
- squared projection onto each full reference and residual norm;
- unweighted row/anchor CE and target token metadata.

For every row in the dominant physical microbatch, record H5/H6 forward RMS
and maxima for the prior `P`, innovation `W`, raw updated memory `Sraw`, raw
successor `Zraw`, stored successor `Z/S`, decoded hidden, and logits. Also
record the boundary RMS-normalization denominators implied by `Sraw` and
`Zraw` before their reuse.

If anchor localization is executed, the anchor contributions must sum back to
the selected row contribution. Token IDs and absolute positions are retained;
no semantic interpretation is inferred from one batch.

## Outputs and success criteria

- `scope_gradients.tsv`
- `gradient_alignments.tsv`
- `row_forward_metrics.tsv`
- `anchor_metadata.tsv`
- `failed_shape_attempt.tsv`
- `interpretation.md`

Metrics remain TSV and interpretation remains Markdown. Checkpoints and tensor
artifacts remain under ignored outputs and are not added to Git.

The audit succeeds only if:

- the four physical-microbatch gradient vectors reconstruct full H5 and full
  H6 with relative L2 error below `5e-4`;
- the 16 row vectors in the selected microbatch reconstruct that microbatch's
  H5 and H6 gradients below `5e-4`;
- if anchor localization runs, its 16 vectors reconstruct the selected row's
  H5 and H6 gradients below `5e-4`;
- all finite/non-finite values and conflicting alignments are retained.

## Evidence boundary

This is a frozen-weight localization of one native batch at step 1000. It does
not reconstruct the historical step-900 spike, establish population
frequency, or identify a training-time normalization/detachment solution.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python eval_byte256_complex_self_predicted_kv_no_qkv_prenorm_step1000_h5_h6_localization.py
```
