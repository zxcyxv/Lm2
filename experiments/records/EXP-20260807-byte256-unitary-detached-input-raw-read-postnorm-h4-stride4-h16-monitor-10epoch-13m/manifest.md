# EXP-20260807 13M detached-hidden recurrent boundary, ten epochs

## Status

- State: stopped after the complete step-1,000 report by user direction to
  rerun with read-only Frobenius normalization; later partial work was not
  registered as a metric row
- Fresh initialization; training and fixed validation splits only; test unread
- Producer: `train_byte256_unitary_detached_input_raw_read_postnorm_h4_stride4_h16_monitor_10epoch_13m.py`

## Question

Does treating the previous central hidden as an external input at each shared
recurrent-block boundary remove the repeated learned hidden-state Jacobian,
while leaving the attached unitary complex-memory path able to train, and does
this 13M model remain finite through ten epoch-equivalents?

## Intervention and comparison

- The previous hidden value is detached before the hidden rotation and Q/K/V
  projections at every central recurrence application. This is a hidden-only
  boundary: the previous complex memory is not detached.
- Memory transport remains a unit-modulus rotation followed by the original
  rank-one complex write. The measurement is the raw linear complex read;
  there is no division by the realized memory Frobenius norm and no
  write-count scaling.
- The measurement projection is added to the rotated hidden. A fixed,
  non-affine RMS normalization of that residual sum is the actual successor
  hidden and the decoder-facing hidden. The initial recurrent root receives
  the same fixed RMS normalization once.
- Q/K/V receive the already post-normalized recurrent state directly, without
  another pre-normalization. There is no memory normalization, damping, gate,
  EMA target, auxiliary loss, or whole-state/TBPTT detach.
- Structural comparison: the 13,215,008-parameter H4/stride-4, H16-monitor
  feedback protocol in
  `EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-10epoch-13m`.
  That configuration keeps the hidden edge attached, keeps its hidden carrier
  raw, and divides the read by realized memory Frobenius norm. Historical
  numeric outcomes are context only, not assumed reproducible evidence for
  this seed run.

## Fixed protocol

- training seed 1337; validation-start seed 2336
- `data/wikitext103_bytes/train.bin`: 55,000,000 byte tokens; SHA-256
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- byte vocabulary 256; context 256; width 1344; two reversible encoder
  blocks; exact inverse decoder; fixed simplex head
- eight complex-memory heads; key dimension 16; value dimension 31;
  exactly 13,215,008 total parameters
- H1--H4 equal token CE only, anchor stride 4; H16 fixed validation, anchor
  stride 16; 256 supervised labels per sequence
- effective batch 64; microbatch 64; 16,384 CE labels per optimizer step
- 33,570 fresh optimizer steps: ten 3,357-step epoch-equivalents by supervised
  label coverage of the 55M-byte training stream
- fused AdamW, betas `(0.9, 0.95)`, zero weight decay, raw global clip norm 1
- strict float32 with TF32 disabled; 100-step linear warmup then cosine decay
  through step 33,570
- fixed 64-example validation; reports at steps 1, 50, 100, 200, 300, 1,000,
  and every 3,357 steps; test split unread
- scalar metrics are written to TSV, interpretation to Markdown, and
  checkpoints stay under ignored `outputs/`

## Success criteria

- Preflight reconstructs both of the first two transitions exactly: zero
  initial memory, raw rank-one writes, raw reads without a realized-norm
  denominator, residual addition, and postnorm successor state.
- A direct boundary probe gives exactly zero gradient to the predecessor
  hidden and a finite nonzero gradient to the predecessor complex memory.
- H4 CE supplies finite nonzero gradients to Q/K/V, readout, hidden and memory
  rotations, and the encoder; total parameter count remains 13,215,008.
- Masked-H16 and truncated-H4 first-four values, loss, and parameter gradients
  agree within the registered strict-float32 tolerances, while H5--H16 logit
  gradients are exactly zero.
- No reported loss or raw gradient is non-finite and no runtime failure occurs.
- Final H1--H4 validation NLL improves from initialization. Transfer is
  recorded, not presumed: mean H5--H16 NLL from epoch 1 to epoch 10 improves
  and at least nine of those twelve horizons improve.
- Any failed criterion or conflicting horizon evidence remains in the record
  with its observed scope.

## Producer command

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_unitary_detached_input_raw_read_postnorm_h4_stride4_h16_monitor_10epoch_13m.py \
  --steps 33570 --schedule-steps 33570 --batch 64 --microbatch 64 \
  --eval-examples 64 --eval-microbatch 16 \
  --record-dir experiments/records/EXP-20260807-byte256-unitary-detached-input-raw-read-postnorm-h4-stride4-h16-monitor-10epoch-13m \
  --output-dir outputs/experiments/EXP-20260807-byte256-unitary-detached-input-raw-read-postnorm-h4-stride4-h16-monitor-10epoch-13m
```
