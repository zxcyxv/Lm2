# EXP-20260802 no-QKV-pre-norm step-900 gradient forensics

## Status

- State: exact reconstruction failed; the historical gradient audit was not
  admitted
- Source run:
  `../EXP-20260802-byte256-complex-self-predicted-kv-urm-boundary-postnorm-no-qkv-prenorm-h16-ce-only-attached-stride16-13m/`
- Source checkpoints: ignored `step0500.pt` and `step1000.pt`
- No native step-900 checkpoint exists. Exact replay must be verified before
  the reconstructed checkpoint is admitted as evidence.
- Updates 501--900 were replayed from `step0500.pt`, but every registered
  verification row failed the preregistered tolerance. The ignored pre/post
  update-900 snapshots therefore belong only to the failed replay trajectory;
  they are not historical step-900 checkpoints.
- Test split remains unmaterialized and unread.

## Question

Which forward tensor, recurrent step, loss horizon, and parameter group first
produce the step-900 raw global gradient norm `1563.952271` and the associated
validation failure? Do horizon detachment or accumulated-read `sqrt(t)`
scaling reduce the instantaneous gradient on the same weights and data?

No mechanism is preregistered as the cause. Large innovation, Q/K/V input
scale, output projection scale, encoder/decoder scale, recurrent Jacobian
products, horizon-gradient alignment, and optimizer state remain competing
hypotheses until decomposed.

## Reconstruction

Resume the exact source model, fused AdamW state, and data-generator state from
step 500. Replay updates 501--900 with the registered loss and schedule. Save
ignored snapshots immediately before and after update 900. Verification rows
at steps 600, 700, 800, and 900 compare replayed train CE and pre-clip global
gradient norm against the preserved source TSV.

Replay is accepted only if each registered scalar agrees within `1e-5`
relative or `1e-6` absolute tolerance. Otherwise all downstream results are
labelled non-exact and cannot identify the historical failure.

## Reconstruction result

| Step | Metric | Source | Replay | Relative error | Accepted |
|---:|---|---:|---:|---:|---|
| 600 | train CE | 3.100221694 | 3.078346074 | 0.007056 | no |
| 600 | pre-clip gradient norm | 26.386745453 | 39.878646851 | 0.511314 | no |
| 700 | train CE | 3.078413010 | 3.153504312 | 0.024393 | no |
| 700 | pre-clip gradient norm | 10.648122787 | 33.773704529 | 2.171799 | no |
| 800 | train CE | 3.065093040 | 3.074492395 | 0.003067 | no |
| 800 | pre-clip gradient norm | 3.379766226 | 6.306803703 | 0.866047 | no |
| 900 | train CE | 10.958056688 | 3.063243568 | 0.720457 | no |
| 900 | pre-clip gradient norm | 1563.952270508 | 2.327549696 | 0.998512 | no |

The replay's largest gradient norm over updates 501--900 was `121.563690`
at step 572. After step 800 its largest norm was `22.286274` at step 834;
there was therefore no post-step-800 threshold crossing to snapshot. The full
conflicting trajectory is retained in `replay_trajectory.tsv`.

The checkpoint payload preserves model, fused AdamW, and the dedicated data
generator, but it does not preserve global CPU/CUDA RNG states, CUDA attention
backend selection, deterministic-algorithm settings, or a checksum of the
mutable producer/common source. The model contains CUDA scaled-dot-product
attention, and the source enabled neither
`torch.use_deterministic_algorithms(True)` nor deterministic cuDNN/SDPA backend
pinning. These are plausible restart differences; the present evidence does
not distinguish them from chaotic amplification of small floating-point
differences. It does establish that `step0500.pt` is insufficient to recreate
the historical transient at the registered tolerance.

## Fixed data and protocol

- WikiText-103 raw UTF-8 training bytes
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- seed 1337 and checkpointed data-generator state
- context 256; window 272; stride 16; H16
- effective batch 64; microbatch 16
- strict float32; TF32 disabled
- source AdamW state, peak LR `3e-4`, fixed 6000-step schedule, clip norm 1
- source loss is the mean over batch, 16 anchors, and 16 horizons

## Decomposition

On the exact pre-update-900 model and exact update-900 batch, record:

1. parameter tensor and module-group weight RMS, spectral norm where defined,
   raw gradient L2/RMS/max, gradient-to-weight ratio, and fraction of squared
   global gradient norm;
2. separate H1--H16 CE gradient vectors, their parameter-group norms, pairwise
   cosine matrix, coherent-sum norm, root-sum-square norm, and cancellation;
3. total-loss gradients for fully attached recurrence and forward-identical
   cross-horizon-detached recurrence;
4. per-step forward RMS/max and retained adjoint RMS/max for root/rotated Z,
   rotated S, Q, prior read, P, K, V, innovation W, raw/stored S, updated Q,
   full read, and raw/stored Z;
5. matrix-free local one-step and H1-to-Hj composed Jacobian spectral estimates
   on a fixed registered probe;
6. module activation and activation-gradient scales across encoder blocks,
   exact inverse decoder calls, and token logits.

Repeat the instantaneous forward/backward audit under four counterfactuals,
without updating weights:

```text
attached, no count scaling                 (historical control)
detached between horizons, no scaling
attached, divide accumulated reads by sqrt(write_count)
detached plus sqrt(write_count)
```

These frozen-weight counterfactuals establish only immediate conditioning.
They do not establish how a model trained from initialization with scaling or
detach would evolve; that requires a separately registered matched retrain.

After-update-900 forward metrics are additionally measured on the first two
registered validation starts to localize the observed validation-scale jump.

## Outputs and success criteria

- `replay_verification.tsv`
- `parameter_gradients.tsv`
- `horizon_gradients.tsv`
- `horizon_gradient_cosines.tsv`
- `recurrent_tensor_metrics.tsv`
- `module_tensor_metrics.tsv`
- `jacobian_metrics.tsv`
- `counterfactuals.tsv`
- `interpretation.md`

Metrics are TSV and interpretation is Markdown. Reconstructed checkpoints and
temporary tensors remain under ignored outputs. Success requires exact replay,
finite diagnostics, squared parameter-group contributions summing to the
reported global norm within numerical tolerance, and the historical attached
control reproducing CE `10.958057` and gnorm `1563.952271`. Conflicting evidence
is retained rather than collapsed into one cause.

## Evidence boundary

The audit localizes the historical step-900 event and evaluates instantaneous
counterfactuals. It does not prove that any counterfactual prevents failure
during a fresh 1000-step training run.

Because reconstruction failed, none of the registered decomposition or
counterfactual claims were produced. The ignored reconstructed snapshots may
be used only to debug replay determinism, not to attribute the source run's
step-900 event.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python replay_byte256_complex_self_predicted_kv_no_qkv_prenorm_step900.py
```
