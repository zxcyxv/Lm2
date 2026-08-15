# EXP-20260802 initial-root RMSNorm anchor-0 byte-h trajectory monitor

## Status

- State: completed; exact step-100/500/1000 candidate-control checkpoint pairs
  were evaluated on the registered fixed validation sample
- Candidate run:
  `../EXP-20260802-byte256-complex-self-predicted-kv-urm-boundary-postnorm-no-qkv-prenorm-initial-root-rmsnorm-h16-ce-only-attached-stride16-13m/`
- Native no-root-normalization control:
  `../EXP-20260802-byte256-complex-self-predicted-kv-urm-boundary-postnorm-no-qkv-prenorm-h16-ce-only-attached-stride16-13m/`
- Causal discovery record:
  `../EXP-20260802-byte256-complex-self-predicted-kv-no-qkv-prenorm-step1000-h5-h6-localization/`
- Test split remains unmaterialized and unread.

## Question

At matched registered checkpoints, does fixed non-affine RMS normalization of
the private initial recurrent root remove the anchor-0 byte-104 (`h`)
trajectory's small H5 pre-boundary hidden scale and subsequent H6 raw-scale
jump, without merely moving the same event to another horizon?

This monitor does not preregister improvement in NLL, gradient norm, or any
particular scale ratio as guaranteed. It measures the intervention's exact
forward trajectory and retains conflicting horizons.

## Fixed sample and checkpoint pairing

- WikiText-103 byte validation split only; train and test are not sampled.
- Validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`.
- Fixed generator seed: `27358` (`1337 + 26021`).
- 256 windows, context 256, window 272, anchor stride 16, H16.
- Physical evaluation microbatch 16, strict float32, TF32 disabled.
- The exact same token windows and batch-16 grouping are used for candidate
  and control.
- Default registered pairs are steps 100, 500, and 1000. The evaluator accepts
  an explicit step list but requires both candidate and control `stepNNNN.pt`
  files for every requested step; it never substitutes `last.pt` or a nearest
  step silently.
- No optimizer, gradient, decoder, token loss, future-token central input, or
  checkpoint mutation is used.

## Groups

Every metric is retained for these disjoint or explicitly overlapping scopes:

- `anchor0_h`: first sparse anchor, current byte ID 104;
- `anchor0_non_h`: first sparse anchor, any other current byte;
- `h_rows_other_anchors`: anchors 1--15 in rows selected by `anchor0_h`;
- `all_anchor_rows`: all 16 anchors in all 256 rows.

`byte_h_counts.tsv` records the exact `anchor0_h` count/rate and asserts that
candidate and control see the same rows. Evaluation requires at least one
byte-h row.

## Recurrent measurements

For H0, record raw encoded-root RMS, initialized recurrent-root RMS, and
initialized memory RMS. For every H1--H16 record rowwise RMS distributions for:

- rotated Z and rotated S;
- decoder-facing prior P;
- rank-one innovation W;
- raw successor Zraw and raw updated memory Sraw;
- pre-boundary Z and S denominators, computed with the model's exact epsilon;
- stored successor Z and S after boundary normalization.

Each summary contains count, mean, median, q90, q99, and maximum. H5/H6 use
the same rows and definitions as every other horizon; they are highlighted in
interpretation rather than measured with a different estimator.

`checkpoint_comparisons.tsv` reports candidate/control ratios and differences
at every matched step, horizon, group, and tensor. It additionally compares
`anchor0_h` against `anchor0_non_h` and `h_rows_other_anchors` within each
model, so a global scale shift cannot masquerade as removal of the byte-h
outlier.

## Cheap training preflight diagnostic

The candidate wrapper should record, without changing its objective:

- `normalize_initial_recurrent_root == True`;
- the initialized private root has per-row RMS one within numerical tolerance;
- its cosine with the raw encoded root is one within numerical tolerance;
- the raw encoder source is unchanged;
- initial memory is generated from the normalized private root and is finite;
- parameter count is unchanged from the native control.

This preflight is structural evidence only. The checkpoint evaluator remains
the registered test of H1--H16 behavior.

## Outputs and completion criteria

- `byte_h_counts.tsv`
- `trajectory_metrics.tsv`
- `checkpoint_comparisons.tsv`
- `interpretation.md`

Metrics remain TSV and interpretation remains Markdown. Checkpoints and raw
tensors remain ignored and are not added to Git.

The monitor completes only when:

- every requested candidate/control checkpoint pair loads strictly;
- byte-h row identities match across the pair;
- every registered group is non-empty;
- all H0/H1--H16 values are finite;
- stored Z/S remain explicitly reported even if raw-scale behavior conflicts
  with the hypothesis.

## Evidence boundary

This is a fixed validation-forward monitoring audit. It does not establish
that initial-root scale caused the historical training spike, and it does not
replace matched NLL, gradient-norm, or fresh-seed training comparisons.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python eval_byte256_complex_self_predicted_kv_initial_root_rmsnorm_anchor0_h_monitor.py \
  --steps 100,500,1000
```

## Result

The fixed sample contained 6 anchor-0 byte-104 rows out of 256. At step 1000,
candidate/control mean ratios on those rows were `0.0687304` for H6 P,
`0.00505199` for H6 innovation W, `0.000829292` for H6 raw successor Z, and
`0.017534` for H6 raw updated S. The control byte-h rows were outliers versus
its non-h rows (H6 W `3.78857x`, Zraw `9.85874x`, Sraw `3.6594x`), while the
candidate ratios were `0.955275x`, `0.975666x`, and `0.987071x`. Across the
candidate's H2--H16 P/W/Zraw/Sraw means, the largest byte-h/non-h ratio was
`1.052613`, so the registered late-horizon event was not merely moved to
another monitored horizon. The candidate also shifted global trajectory
scales downward, so this audit supports removal of the localized event but
does not by itself identify a unique causal pathway.
