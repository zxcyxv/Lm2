# EXP-20260805 13M feedback recurrence, H1 loss with H16 monitoring

## Status

- State: two-epoch training completed at step 6,714; corrected greedy
  self-reinput block evaluation pending
- Fresh initialization; training and fixed validation splits only; test unread
- Structural parent:
  `EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-normalized-residual-h16-ce-only-attached-stride16-13m`
- Training completed in 845.777 seconds with finite final H1 validation NLL
  `1.303748174`; the step-6,714 checkpoint is under ignored `outputs/`.
- The inherited summary also reports open-H16 checks that were registered
  before the user clarified the intended self-reinput metric. Those rows are
  retained as out-of-contract diagnostics and are not the outcome criterion.

## Question and hypothesis

After the epoch-2 H4-loss result, does optimizing only the first
state-dependent central transition and re-entering every greedy selected
token produce a lower sixteen-token rollout CE than the H4-trained model's
natural four-token blocks? Gold context is restored only at each sixteen-token
evaluation boundary. This natural chunk-size comparison at step 6,714 is the
primary metric. Open-loop H16 is recorded by the inherited monitor but lies
outside the H1 training/inference contract and is not a model-performance
ranking metric.

## Intervention and comparisons

- Keep the 13,215,008-parameter width-1344 unitary branch-normalized feedback
  architecture unchanged.
- Backpropagate equal CE from H1 only at 256 stride-one anchors. This supplies
  exactly `256 * 1 = 256` labels per sequence, matching H4/stride-4 and
  H16/stride-16 without loss rescaling.
- Sample the same 260-token training windows as H4 using the same generator;
  the H1 objective consumes positions 0--256 and ignores the trailing three.
  Thus both arms supervise the identical 256 future corpus bytes from
  identical sampled starts, rather than merely matching label counts.
- Validation records the inherited open-H16 diagnostic at sixteen stride-16
  anchors, without treating it as the H1 model's intended inference path.
- Primary natural-chunk control: epoch-2 H4/stride-4 feedback checkpoint and record
  `EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-10epoch-13m`.
  Its step-6,714 open H1--H16 NLL is `3.616869517`.
- Secondary chunk-size control: the corrected audit can run both checkpoints
  with greedy one-token re-entry, while keeping the requested natural H1
  token-one versus H4 block-four comparison primary.
- Epoch-2 parallel-scan open H1--H16 NLL `2.912845770` is retained as
  historical context only; it is not conditioning-matched to token re-entry.

## Fixed protocol

- seed 1337; validation-start seed 2336
- `data/wikitext103_bytes/train.bin`, 55,000,000 byte tokens, SHA-256
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- byte vocabulary 256; context 256; width 1344; two reversible encoder
  blocks; exact inverse decoder; fixed simplex head
- eight heads; complex key dimension 16; value dimension 31
- raw latent and memory carriers; fixed non-affine RMS on QKV and decoder
  branches; Frobenius-normalized measurement; no damping, detach, EMA,
  auxiliary loss, or token re-encoding in training
- H1/stride-1 training; open H16/stride-16 fixed validation
- effective batch = microbatch = 64; 16,384 CE labels per optimizer step
- 6,714 fresh optimizer steps: two 3,357-step label-coverage epochs
- fused AdamW, beta `(0.9, 0.95)`, zero weight decay, raw global clip norm 1.0
- strict float32, TF32 disabled; 100-step linear warmup then the same
  33,570-step cosine schedule used by H4. Training stops at step 6,714, so the
  first-two-epoch LR trajectory is exactly matched
- fixed 64-example validation; reports at 1/50/100/200/300/1000 and both
  epoch boundaries; test split unread
- scalar metrics in TSV, interpretation in Markdown, checkpoints under ignored
  `outputs/`

## Success and interpretation criteria

- parameter count remains exactly 13,215,008; Q/K/V, readout, hidden phase,
  and encoder gradients must be finite and nonzero. The cross-horizon memory
  phase must have exactly zero H1 gradient because initial memory is zero and
  this parameter first acts after H1; preserve that structural zero as
  evidence rather than treating it as a runtime failure
- preflight proves an explicit H16 graph masked to H1 matches the causally
  truncated H1 graph in selected forward values, loss, and representative
  parameter gradients; H2--H16 logit gradients must be exactly zero
- no non-finite loss, gradient, or validation metric and no runtime failure
- H1 NLL improves from initialization
- primary outcome: compare H1-loss greedy-token-one sixteen-token rollout CE
  with H4-loss greedy-block-four rollout CE, restoring gold context only at
  sixteen-token boundaries
- secondary outcome: compare H1 and H4 under identical greedy-token-one
  re-entry to help isolate the training-horizon effect
- retain the inherited open-H16 trace as an explicitly out-of-contract
  diagnostic; do not use it to rank the H1 token-reinput model

## Preflight (not outcome evidence)

- Model size was 13,215,008 parameters, 12,869,600 trainable.
- Explicit masked-H16 versus truncated-H1 checks measured first-step forward
  max error `3.433e-5`, selected-loss absolute error `3.815e-6`, and
  representative parameter-gradient relative error `2.357e-5`.
- H2--H16 logit-gradient max and H1 memory-phase gradient were exactly zero.
  H1 Q/K/V/readout/hidden-phase/encoder gradient norms were respectively
  `4.494`, `4.701`, `4.173`, `2.589`, `0.683`, and `58.544`.
- Keeping the H4-matched 260-token sampled window while slicing the H1
  objective to 257 positions changed the H1 loss by exactly zero.
- A batch-64/microbatch-64 update completed with finite CE `48.1877`, raw
  gradient norm `123.953` before clipping, and peak allocated VRAM
  8,289,438,208 bytes. Smoke records/checkpoints are under `/tmp`, not Git.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_unitary_feedback_h1_stride1_h16_monitor_2epoch.py \
  --steps 6714 --schedule-steps 33570 --batch 64 --microbatch 64 \
  --eval-examples 64 --eval-microbatch 16 \
  --record-dir experiments/records/EXP-20260805-byte256-unitary-feedback-h1-stride1-h16-monitor-2epoch-13m \
  --output-dir outputs/experiments/EXP-20260805-byte256-unitary-feedback-h1-stride1-h16-monitor-2epoch-13m
```
