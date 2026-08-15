# EXP-20260805 13M feedback recurrence, H4 loss with H16 monitoring

## Status

- State: stopped at step 13,428 (four epoch-equivalents) by explicit user
  direction in order to run the H1-only comparison
- Fresh initialization; training and fixed validation splits only; test unread
- Structural parent:
  `EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-normalized-residual-h16-ce-only-attached-stride16-13m`
- The step-13,428 checkpoint was preserved under ignored `outputs/`; the
  originally registered ten-epoch success criteria were not evaluated as if
  this curtailed run had completed.

## Question and hypothesis

When the shared state-dependent central transition is optimized only through
H1--H4, do improvements learned at shallow horizons transfer outward to the
untrained H5--H16 rollout?  The motivating hypothesis is a sequential
curriculum: a better H1 transition makes H2 easier, and improving shallow
transitions can subsequently increase the rate at which deeper-horizon NLL
falls.

This is not the weaker claim that every horizon's NLL curve is merely convex
while decreasing.  The analysis will distinguish each interval's positive
improvement velocity

```math
v_h(e)=L_h(e-1)-L_h(e)
```

from changes in that velocity, and will examine whether shallow improvement
precedes deeper improvement.  Epoch-level traces are observational evidence;
they do not by themselves identify a causal edge between horizons.

## Intervention and comparisons

- Keep the 13,215,008-parameter width-1344 unitary branch-normalized
  state-dependent feedback architecture unchanged.
- The registered objective is the H16 feedback graph with equal token CE
  selected only from H1--H4. H5--H16 logits have exactly zero loss gradient.
  Because the inverse tape and recurrence are causal, H5--H16 are descendants
  of H1--H4 and cannot affect that selected loss. The producer therefore
  dead-code-eliminates those unused training nodes after preflight verifies
  that the first-four logits, selected loss, and parameter gradients match an
  explicit masked H16 graph. Validation always executes the full H16 graph.
- Use 64 stride-four anchors, so each sequence still supplies exactly
  `64 * 4 = 256` CE labels.  This matches the label count and byte coverage of
  the H16/stride-16 runs (`16 * 16 = 256`) without rescaling the loss.
- During validation, unfold the same learned feedback transition through H16
  at stride-16 anchors. H5--H16 are out-of-objective conditional-depth
  transfer metrics. They are not held-out token identities: under stride 4,
  the same corpus position can be an H1--H4 label from a later anchor.
- Primary historical comparison: the completed ten-epoch time-varying
  parallel-scan record
  `EXP-20260803-byte256-unitary-fused-triton-scan-h16-10epoch-13m-rtx5090`.
  That comparison changes both the feedback edge and supervised horizon and
  therefore is descriptive, not a single-factor architectural ablation.
- Secondary context: the completed 6,000-step fully supervised H16 feedback
  continuation.  Its shorter schedule is not an epoch-matched control.

## Fixed protocol

- seed 1337; validation-start seed 2336
- `data/wikitext103_bytes/train.bin`, 55,000,000 byte tokens, SHA-256
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- byte vocabulary 256; context 256; width 1344; two reversible encoder
  blocks; exact inverse decoder; fixed simplex head
- state-dependent unitary feedback recurrence; eight heads; complex key
  dimension 16; value dimension 31
- raw latent and memory carriers; fixed non-affine RMS on QKV and decoder
  branches; Frobenius-normalized measurement; no damping, detach, EMA,
  auxiliary loss, or token re-encoding
- masked-H16-equivalent H1--H4 training, anchor stride 4; H16 fixed
  validation, anchor stride 16
- effective batch = microbatch = 64; 16,384 CE labels per optimizer step
- 33,570 fresh optimizer steps: ten 3,357-step epoch-equivalents by supervised
  label coverage of the 55M-byte stream
- fused AdamW, beta `(0.9, 0.95)`, zero weight decay, raw global clip norm 1.0
- strict float32, TF32 disabled; 100-step linear warmup then cosine decay over
  all 33,570 steps
- fixed 64-example validation; reports at 1/50/100/200/300/1000 and every
  3,357 steps; test split unread
- scalar metrics go to TSV, interpretation to Markdown, and checkpoints stay
  under ignored `outputs/`

## Success and interpretation criteria

- parameter count remains exactly 13,215,008; preflight must prove finite,
  nonzero H4 CE gradients reach Q/K/V, readout, rotations, and encoder
- preflight must also prove that masked H16 and causally truncated H4 have
  matching first-four forward values, loss, and parameter gradients, while
  H5--H16 logit gradients are exactly zero. Strict-float32 tape-length
  association order may differ; the registered maximum relative parameter
  gradient tolerance is `2e-4` (the first observed smoke value was
  `8.18131e-5` under the otherwise passing checks)
- no non-finite reported loss or gradient and no runtime failure
- H1--H4 final NLL must improve from initialization
- transfer criterion: from epoch 1 to epoch 10, mean H5--H16 NLL improves and
  at least 9 of the 12 held-out horizons improve
- report, without deleting counterevidence, every H1--H16 epoch NLL and
  interval improvement velocity
- assess concurrent and one-epoch-lag shallow/deep velocity association,
  shallow-to-deep ordering of peak improvement intervals, and the reverse-lag
  comparison; with only ten epoch intervals these are descriptive and must
  not be reported as proof of causation

## Preflight (not outcome evidence)

- System Python used the existing PyTorch `2.8.0+cu128`; the local project was
  installed editable with `python -m pip install -e . --no-deps`, so no
  dependency or PyTorch version was changed.
- Model size was exactly 13,215,008 parameters, of which 12,869,600 were
  trainable.
- Masked-H16 versus causally truncated-H4 checks measured first-four forward
  max error `3.81470e-5`, selected-loss absolute error `0`, and parameter
  gradient max relative error `8.18131e-5`. H1--H4 logit-gradient norm was
  `0.0855817`; H5--H16 logit-gradient max was exactly `0`.
- A batch-64/microbatch-64 update completed with finite CE `46.3012`, raw
  gradient norm `121.702` before clip, and peak allocated VRAM 7,787,565,056
  bytes. A separate H16 validation microbatch-16 smoke also completed.
- Smoke checkpoints and raw smoke records were written only under `/tmp` and
  are not part of the repository evidence record.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch.py \
  --steps 33570 --schedule-steps 33570 --batch 64 --microbatch 64 \
  --eval-examples 64 --eval-microbatch 16 \
  --record-dir experiments/records/EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-10epoch-13m \
  --output-dir outputs/experiments/EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-10epoch-13m
```

After all ten epoch rows exist, derive the registered transfer and lead--lag
tables without reading checkpoints or the test split:

```bash
python analyze_byte256_feedback_h4_h16_transfer.py
```

This writes metric tables `epoch_horizon_nll.tsv`, `interval_metrics.tsv`,
`group_interval_metrics.tsv`, `horizon_summary.tsv`, `cascade_summary.tsv`,
and `scan_comparison.tsv` plus the separate Markdown interpretation
`transfer_analysis.md`.
