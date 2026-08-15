# EXP-20260802 rotated-hidden innovation residual H16

## Status

- State: completed; all 300 updates were finite, but the preregistered
  gradient-stability criterion failed
- Direct matched control:
  `../EXP-20260802-byte256-complex-self-predicted-kv-urm-boundary-postnorm-no-qkv-prenorm-initial-root-rmsnorm-h16-ce-only-attached-stride16-13m/`
- The completed main producer is the control. Its preserved interrupted first
  attempt is not used as the comparison.
- Test split remains unmaterialized and unread.

## Question

Does an explicit same-level unitary hidden residual stabilize the fully
attached H16 recurrence through step 300 without suppressing likelihood
learning?

## Sole intervention

```text
control:   Znext = fixed_RMS(full_read(Q(P), U S + write(P)))
candidate: Znext = fixed_RMS(R Z + innovation_read(Q(P), write(P)))
```

The candidate carries the rotated previous hidden directly and adds only the
new rank-one innovation read as a delta. It does not add the absolute full
updated-memory read to the carrier. The CE-facing P, updated memory
`U S + write(P)`, and every parameter are unchanged.

Everything else is inherited from the direct control: private initial-root
fixed RMSNorm, no QKV pre-normalization, final-carrier-only fixed Z/S RMSNorm,
no accumulated-read count scaling, `residual_prior=False`, exact inverse
decoder, raw simplex-tied head scale 16, H16 fully attached stride-16 CE, and
no beta, auxiliary loss, detach, EMA, token feedback, or root reinjection.

## Fixed protocol

- WikiText-103 raw UTF-8 train and validation bytes only
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed 2336
- context 256; window 272
- 16 stride-16 anchors by 16 horizons = 256 CE labels per sequence
- effective batch 64; physical microbatch 16; four-way accumulation
- strict float32; TF32 disabled
- AdamW, betas `(0.9, 0.95)`, weight decay 0, peak LR `3e-4`
- clip norm 1.0; recorded `gradient_norm` is raw pre-clip global L2
- 300 updates on the unchanged 6000-update schedule
- reports at 0, 1, 50, 100, 200, and 300
- 64 fixed validation examples; evaluation microbatch 2
- online model only

## Required preflight

Before optimizer update, the producer must verify and record:

- recurrence mode is `rotated-hidden-innovation-residual`;
- raw successor is exactly `RZ + innovation_delta` and stored successor is
  exactly its fixed non-affine RMS normalization;
- the candidate differs from an absolute full updated-memory read;
- P remains residual-free and the inherited root/QKV/boundary configuration
  is unchanged;
- parameter count remains 13,215,008 total and 12,869,600 trainable;
- future-input independence, causal inverse triangularity, exact RevNet
  inverse, complex phases/readout, and fully attached finite H16 gradients
  pass the inherited checks.

## Metrics and diagnostics

The producer retains the matched TSV schema: train CE, raw pre-clip global
gradient norm, block and H1--H16 validation NLL/accuracy, horizon P and stored
Z RMS, memory and innovation energy, logit RMS, throughput, and VRAM. Metrics
remain TSV and interpretation remains Markdown. Checkpoints stay under ignored
`outputs/experiments/`.

The direct control prefix is fixed as follows:

| Step | Control gnorm | Control block NLL |
|---:|---:|---:|
| 1 | 215.868988 | 6.129311 |
| 50 | 21.100031 | 3.353405 |
| 100 | 68.031082 | 3.558278 |
| 200 | 66.590050 | 3.175779 |
| 300 | 49.031799 | 3.163019 |

## Comparisons and success criteria

The step-300 stability criterion passes only if:

- all 300 updates and every registered validation row are finite;
- rows 1, 50, 100, 200, and 300 exist with finite train loss, raw gnorm,
  block NLL, and horizon scale diagnostics;
- candidate gnorm is below the direct control at every one of those rows;
- candidate raw gnorm is at most `10.0` at steps 50, 100, 200, and 300;
- step-300 block NLL improves from initialization, is below `4.0`, and is no
  more than `0.15` worse than the direct control (`<= 3.313019`);
- every H1--H16 step-300 NLL improves from initialization;
- the sole-intervention preflight passes without relaxing inherited checks.

No monotonic validation-NLL criterion is imposed: the direct control itself
has a registered step-100 bump. Gradient clipping remains enabled and cannot
be cited as proof of raw-gradient stability because the recorded norm is
pre-clip.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_rotated_hidden_innovation_residual_h16_attached_stride16_ce_only_13m.py \
  --steps 300 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Evidence boundary

This run tests only the explicit hidden residual through 300 updates. It does
not test root reinjection, a learned or horizon-scaled residual gate, a
baseline-subtracted full read, count scaling, truncation, H32/H128, test
performance, or autonomous-generation quality.

## Result

The exact formula and inherited preflight checks passed, and all registered
training and validation rows remained finite. Likelihood learning was
preserved: block NLL improved from `6.854815` to `3.203612`, every H1--H16 NLL
improved from initialization, and the final block NLL stayed within the
registered `0.15` tolerance of the matched control's `3.163019`.

The hidden residual did not meet the stability criterion:

| Step | Candidate gnorm | Matched gnorm | Candidate block NLL | Matched block NLL |
|---:|---:|---:|---:|---:|
| 1 | 235.902512 | 215.868988 | 6.499490 | 6.129311 |
| 50 | 53.555164 | 21.100031 | 3.509794 | 3.353405 |
| 100 | 35.028534 | 68.031082 | 3.453836 | 3.558278 |
| 200 | 16.550217 | 66.590050 | 3.182449 | 3.175779 |
| 300 | 33.470047 | 49.031799 | 3.203612 | 3.163019 |

It reduced the matched gradient at steps 100, 200, and 300, but exceeded the
control at steps 1 and 50, never reached the registered post-warmup ceiling of
10, and rebounded from `16.550217` at step 200 to `33.470047` at step 300.
Therefore a bare unitary-carrier plus innovation-only residual is insufficient
to call the H16 recurrence gradient-stable.
