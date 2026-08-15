# EXP-20260802 initial recurrent-root RMSNorm H16

## Status

- State: interrupted after the registered step-300 report; preserved read-only
  as recovery evidence after the execution environment was recreated
- Parent metrics SHA-256 at preservation:
  `f868ce18821c9700ecd6ffe54d3a134871a28d52c477b96287ac2860b0bdb2e6`
- Ignored `last.pt` SHA-256 at preservation:
  `a1b4ff8f91cdc1c7863c39240a556e946e3054c6f568963bacd445e3cba32261`
- Exact matched control:
  `../EXP-20260802-byte256-complex-self-predicted-kv-urm-boundary-postnorm-no-qkv-prenorm-h16-ce-only-attached-stride16-13m/`
- Test split remains unmaterialized and unread.

## Question

Does fixed non-affine RMS normalization of only the initial recurrent root
remove the poorly scaled H1 entry regime and stabilize the H5/H6 sensitivity
event, while preserving the no-QKV-pre-norm boundary-postnorm H16 model and
its exact RevNet inverse decoder?

## Sole intervention

```text
control:   Z0 = encoder_root;              S0 = write(encoder_root)
candidate: Z0 = fixed_RMS(encoder_root);   S0 = write(Z0)
```

`fixed_RMS` has no affine parameters and uses the already registered
`post_norm_eps=1e-6`. The encoder output itself is not overwritten; only the
private root passed into initial recurrent Z/S construction is normalized.
Parameter count is unchanged.

Everything after initialization is held fixed from the matched control:
central Q/K/V receive raw inputs through exact `Identity` with no QKV pre-norm;
P, innovation W, raw memory update, and raw successor read remain unnormalized;
only the complete successor Z/S carrier receives fixed non-affine boundary
normalization before reuse. H16 remains fully attached with stride-16 anchors
and equal CE on all 16 horizons. There is no count scaling, detach, MSE, KL,
EMA, beta, token feedback, vocabulary-head change, or RevNet path change.

## Fixed protocol

- WikiText-103 raw UTF-8 train and validation bytes only
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed `1337 + 999`
- context 256; window 272
- train/evaluation anchor stride 16
- 16 anchors by 16 horizons = 256 CE labels per sequence
- effective batch 64; physical microbatch 16
- strict float32; TF32 disabled
- AdamW, peak LR `3e-4`, clip norm 1.0
- 1000 updates on the fixed 6000-update schedule
- reports at 0, 1, 50, 100, then every 100 through 1000
- 64 validation examples, evaluation microbatch 2
- online model only

## Required preflight

Before any optimizer update, the producer must verify and record:

- `normalize_initial_recurrent_root` is true, while QKV pre-normalization is
  false and its module is exact `Identity`;
- initialized Z0 equals the fixed non-affine RMS formula applied to the raw
  encoder root, has finite per-row RMS near one, and differs only by a positive
  rowwise scalar (cosine near one);
- initialized S0 equals `write(Z0)` and is finite;
- the intervention adds no parameters;
- the inherited future-input independence, causal inverse-decoder
  triangularity, complex phases/readout, finite H16 gradients, and exact
  RevNet inverse path all still pass.

## Metrics and H5/H6 diagnostics

The training producer keeps the matched global metrics schema: H1--H16 NLL and
accuracy, block NLL, raw pre-clip global gradient norm, per-horizon P/Z/memory,
innovation energy and logit scales, VRAM, and elapsed time. In particular,
steps 50, 100, 300, 500, 900, and 1000 must retain the global gradient norm and
all H5/H6 NLL, P RMS, recurrent-Z RMS, innovation energy, memory energy, and
logit RMS fields even when evidence conflicts.

A separate post-run checkpoint evaluator will perform targeted byte-anchor
H5/H6 gradient localization and compare the candidate with native matched
control checkpoints. That secondary audit is required evidence but is not a
training-producer success gate and must not be substituted with global gnorm.

Metrics are TSV and interpretation is Markdown. Checkpoints and smoke
artifacts remain under ignored outputs.

## Comparisons and success criteria

Primary matched success requires:

- all 1000 updates and every registered validation row remain finite;
- final block validation NLL is below `4.0`;
- all H1--H16 final NLL values improve from initialization;
- the required step 50/100/300/500/900/1000 rows exist, with finite raw global
  gnorm and finite H5/H6 scale diagnostics;
- the sole-intervention preflight passes without relaxing the matched
  architecture checks.

The matched no-initial-normalization control and all step-900/step-1000
forensics remain preserved. Lower H5/H6 scales or gnorm are supporting evidence,
not sufficient success by themselves; final likelihood cannot be replaced by
scale reduction.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_urm_boundary_postnorm_no_qkv_prenorm_initial_root_rmsnorm_h16_attached_stride16_ce_only_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Evidence boundary

This isolates only initial private recurrent-root RMS normalization. It does
not test intermediate normalization, accumulated-read scaling, innovation
gating, detachment, a different optimizer/LR, H32/H128, AR equivalence, or test
performance.
