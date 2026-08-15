# EXP-20260807 13M full-BPTT joint postnorm, ten epochs

## Status

- State: registered before execution
- Fresh initialization; training and fixed validation splits only; test unread
- Producer:
  `train_byte256_unitary_full_bptt_raw_read_joint_postnorm_h4_stride4_h16_monitor_10epoch_13m.py`

## Question

With all recurrent gradients attached, is fixed non-affine post-normalization
of both fields of the complete central carry sufficient to keep the H4-trained
13M recurrence finite and trainable? In particular, does normalizing the
updated complex memory before the raw read prevent forward memory growth from
appearing as a growing local query-gradient scale without truncating
future-to-past credit assignment?

## Intervention and comparison

- The predecessor hidden and predecessor complex memory both remain attached
  at every central application. The inverse-decoder branch tape also remains
  attached. This is full BPTT through the registered H4 training graph.
- The private initial recurrent hidden receives fixed non-affine RMS
  normalization once and memory starts at exactly zero.
- Every central use computes unitary memory transport plus the original
  rank-one complex write. The complete updated memory receives one global
  fixed non-affine complex RMS post-normalization over head, key, and value
  axes before it is read and before it becomes the next memory carrier.
- The read remains the raw linear `q^dagger S/sqrt(key_dim)` measurement. No
  realized memory norm appears in a separate read denominator and there is no
  write-count scaling.
- The measurement projection is added to the unitary-rotated hidden. Fixed
  non-affine RMS normalization of this sum is both the decoder-facing hidden
  and the next hidden carrier.
- Q/K/V consume the already normalized hidden directly. There is no extra
  QKV pre-normalization, damping, gate, EMA target, or auxiliary loss.
- Primary structural control:
  `EXP-20260807-byte256-unitary-detached-input-raw-read-postnorm-h4-stride4-h16-monitor-10epoch-13m`,
  which has the same hidden postnorm and raw read but detaches the predecessor
  hidden and leaves memory unnormalized. This requested C arm changes both
  detach and memory normalization relative to that control; it does not by
  itself identify either intervention's isolated causal effect.

## Fixed protocol

- optimization seed 1337; validation-start seed 2336
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
- fixed 64-example validation; reports at steps 1 and 50, then every 100 steps,
  with all epoch boundaries included; test split unread
- scalar metrics go to TSV, interpretation to Markdown, and checkpoints stay
  under ignored `outputs/`

## Registered measurements and success criteria

- Preflight must reconstruct the first two transitions from the literal
  equations: zero initial memory, unitary transport, raw rank-one write,
  complex-memory RMS postnorm before raw read, residual addition, and hidden
  RMS postnorm.
- The first two successor hidden and memory RMS values must lie in `(0.99, 1]`.
- An independent boundary probe must give finite nonzero gradients to both the
  predecessor hidden and predecessor complex memory.
- Masked-H16 and truncated-H4 first-four values, loss, and parameter gradients
  must agree within the existing strict-float32 tolerances; H5--H16 logit
  gradients must be exactly zero in the training objective.
- Every reported training loss, validation loss, and raw pre-clip global
  gradient norm must be finite. Absolute global gradient norm is recorded but
  is not alone classified as recurrent Jacobian amplification because four CE
  losses and four tied parameter uses contribute additively.
- H1--H4 validation NLL must improve from initialization. H5--H16 transfer is
  recorded under the inherited monitor criteria and is not assumed.
- A later H4-only recurrent-adjoint audit may localize amplification, but no
  such result is claimed by this training manifest before execution.

## Command

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_unitary_full_bptt_raw_read_joint_postnorm_h4_stride4_h16_monitor_10epoch_13m.py \
  --steps 33570 --schedule-steps 33570 --batch 64 --microbatch 64 \
  --eval-examples 64 --eval-microbatch 16 \
  --record-dir experiments/records/EXP-20260807-byte256-unitary-full-bptt-raw-read-joint-postnorm-h4-stride4-h16-monitor-10epoch-13m \
  --output-dir outputs/experiments/EXP-20260807-byte256-unitary-full-bptt-raw-read-joint-postnorm-h4-stride4-h16-monitor-10epoch-13m
```
