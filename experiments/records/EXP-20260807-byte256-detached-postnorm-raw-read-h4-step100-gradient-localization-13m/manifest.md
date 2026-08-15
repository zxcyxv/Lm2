# EXP-20260807 detached-postnorm step-100 gradient localization

## Status

- State: complete; fresh optimizer step 100 and the registered fixed-batch
  gradient audit were produced
- Fresh initialization; training split for optimization and the fixed
  diagnostic batch, validation split for ordinary reports; test unread
- Training producer:
  `train_byte256_detached_postnorm_raw_read_h4_step100_gradient_localization_13m.py`
- Gradient producer:
  `analyze_byte256_detached_postnorm_step100_gradients.py`

## Recorded outcome

- The optimizer batch at step 100 had raw pre-clip global gradient norm
  `102.742355`; the separately seeded fixed diagnostic batch had
  `89.525576` for the equal H1--H4 mean CE.
- Causal reach was exactly triangular: H1 reached central use 1; H2 uses
  1--2; H3 uses 1--3; and H4 all four uses. All future-use gradients were
  exactly zero and all causally reachable use-group norms were finite and
  nonzero.
- For H4, use-specific central parameter norms were `24.475891`,
  `18.965165`, `15.815067`, and `76.805710` from uses 1 through 4. The H4
  successor-memory adjoint varied by only `0.1079%` across the four uses.
- The tied and audit-untied logits were bit-identical. The maximum relative
  max-element error between a tied central gradient and the sum of its four
  use-specific gradients was `1.9858e-7`.
- Detailed measurements are in the registered TSV files; interpretation and
  scope limits are in `gradient_analysis.md`.

## Question

At step 100 of the 13M detached-hidden, raw-read, recurrent-postnorm H4 model,
which parameter layers and which of the four shared central-transition uses
account for the pre-clip gradient norm? For each separate H1, H2, H3, and H4
CE, does gradient reach exactly the causally available recurrent prefix, and
does its magnitude systematically increase toward earlier recurrent uses?

## Intervention and comparison

- Architecture and objective are unchanged from
  `EXP-20260807-byte256-unitary-detached-input-raw-read-postnorm-h4-stride4-h16-monitor-10epoch-13m`.
- Previous hidden values are detached at central block entry; complex memory
  remains attached; the complex read is raw; residual-plus-readout is the
  fixed-RMS-normalized successor hidden.
- Equal H1--H4 CE is trained at stride-four anchors. The learning-rate schedule
  remains the registered 33,570-step schedule, but this fresh run stops after
  optimizer step 100.
- The audit evaluates one fixed 64-example training-split batch at the saved
  step-100 weights, accumulated in eight-example diagnostic microbatches. It
  computes H1, H2, H3, H4 CE separately and their equal mean.
- For audit only, the shared central parameters are cloned into four
  value-identical per-use parameter sets. Forward logits must remain equal to
  the ordinary tied rollout. Per-use gradients are recorded separately, and
  their parameter-wise sum must reproduce the ordinary tied gradient.

## Fixed protocol

- optimization seed 1337; validation-start seed 2336; diagnostic-batch seed
  44437 (`1337 + 43100`)
- `data/wikitext103_bytes/train.bin`: 55,000,000 byte tokens; SHA-256
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- byte vocabulary 256; context 256; width 1344; two reversible encoder
  blocks; exact inverse decoder; fixed simplex head
- eight complex-memory heads; key dimension 16; value dimension 31;
  13,215,008 total parameters and 12,869,600 trainable parameters
- H1--H4 mean CE, 64 stride-four anchors, effective batch and microbatch 64
- fused AdamW, betas `(0.9, 0.95)`, zero weight decay, global clip norm 1;
  strict float32 and TF32 disabled
- 100 optimizer steps on the unchanged 33,570-step warmup/cosine schedule;
  ordinary reports at steps 0, 1, 50, and 100
- diagnostic gradients are never passed to an optimizer
- scalar and tensor-gradient measurements go to TSV; interpretation goes to
  Markdown; checkpoints remain under ignored `outputs/`

## Registered measurements and success criteria

- `parameter_gradients.tsv`: every trainable named parameter under separate
  H1--H4 and mean-H4 objectives
- `layer_gradient_norms.tsv`: encoder attention/FFN groups, the tied central
  transition, and complete-model global norms
- `central_use_parameter_gradients.tsv`: Q/K/V/output/phase gradients for each
  of the four value-identical central uses
- `recurrent_tensor_gradients.tsv`: per-loss, per-step gradients of rotated
  hidden, measurement delta, successor hidden, write, and successor memory
- `gradient_consistency.tsv`: forward equality and equality between the tied
  central gradient and the sum of four untied-use gradients
- all values must be finite; H1 must not reach future central uses; H4 must
  reach every causally available use through the decoder and/or live memory
- no claim that upstream growth is pathological is made from monotonicity
  alone; ordinary accumulation, vector alignment, and Jacobian amplification
  are distinguished in the interpretation

## Module-VJP amplification follow-up

- Registered before execution after the parameter-gradient audit proved
  insufficient to locate amplification across computation boundaries.
- Reuse `step0100.pt`; do not retrain and do not alter model code or forward
  equations.
- Replay the first eight examples of the already registered seed-44437
  diagnostic batch. Compute H1, H2, H3, H4, and mean CE separately.
- Attach backward hooks to every executed encoder/inverse-decoder RMSNorm,
  attention, QKV/projection, FFN linear/GELU/composite, reversible-block, and
  central output-projection call. Record input/output adjoint RMS and the local
  VJP RMS gain for every call in `central_operation_vjp_gains.tsv`.
- Record logits, decoded hidden, encoded prefix, recurrent hidden/read/write,
  and memory activation adjoints by recurrence use in
  `central_activation_adjoint_norms.tsv`.
- All reported adjoints and gains must be finite. Results identify a local
  amplification site only from an input/output adjoint ratio at the same
  executed operation, not from a parameter-gradient norm.

## Commands

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_detached_postnorm_raw_read_h4_step100_gradient_localization_13m.py \
  --steps 100 --schedule-steps 33570 --batch 64 --microbatch 64 \
  --eval-examples 64 --eval-microbatch 16 \
  --record-dir experiments/records/EXP-20260807-byte256-detached-postnorm-raw-read-h4-step100-gradient-localization-13m \
  --output-dir outputs/experiments/EXP-20260807-byte256-detached-postnorm-raw-read-h4-step100-gradient-localization-13m

python analyze_byte256_detached_postnorm_step100_gradients.py \
  --checkpoint outputs/experiments/EXP-20260807-byte256-detached-postnorm-raw-read-h4-step100-gradient-localization-13m/step0100.pt \
  --record-dir experiments/records/EXP-20260807-byte256-detached-postnorm-raw-read-h4-step100-gradient-localization-13m \
  --batch 64 --microbatch 8
```
