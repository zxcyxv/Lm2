# EXP-20260802 no-QKV-pre-norm step-1000 one-update forensics

## Status

- State: completed; native step-1000 one-update audit executed successfully
- Source run:
  `../EXP-20260802-byte256-complex-self-predicted-kv-urm-boundary-postnorm-no-qkv-prenorm-h16-ce-only-attached-stride16-13m/`
- Source checkpoint: ignored native `step1000.pt`
- The failed attempt to reconstruct historical step 900 is preserved in:
  `../EXP-20260802-byte256-complex-self-predicted-kv-no-qkv-prenorm-step900-gradient-forensics/`
- Test split remains unmaterialized and unread.

## Question

For one controlled optimizer update from the native step-1000 checkpoint,
where does the raw gradient enter, how does it propagate across H1--H16 and
the internal layers, how much does clipping plus Adam actually move each
parameter group, and which forward tensor scales change after that movement?

The audit does not preregister QKV scale, innovation, recurrent Jacobian,
encoder/decoder sharing, horizon-gradient alignment, detach, or `sqrt(t)` as
the cause or solution.

## Fixed update

- Load the native source `step1000.pt` model and fused AdamW state.
- Restore its checkpointed training-data generator.
- Draw exactly the next batch: effective batch 64, microbatch 16, context 256,
  window 272, stride 16, H16.
- Use update index 1000 and the source 6000-step LR schedule.
- Source objective: mean CE over batch, 16 anchors, and 16 horizons.
- Record gradients before clip norm 1, the clipped gradients, Adam state, and
  exact parameter delta after `optimizer.step()`.
- Do not continue beyond this single update.

Train SHA-256:
`062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`.
Strict float32 is used with TF32 disabled. Seed and generator state come from
the checkpoint.

## Required decomposition

1. Every trainable parameter tensor and grouped encoder block, central Q/K/V/O,
   phases, inverse-decoder/shared block use: weight RMS/L2/max, raw gradient
   RMS/L2/max, squared-global-norm share, clipped gradient, Adam update L2/RMS,
   update-to-weight ratio.
2. H1--H16 loss gradients separately, scaled by the historical `1/16` factor,
   plus their pairwise cosine, coherent sum, root-sum-square, and cancellation.
3. Each recurrent occurrence: input/rotated/stored Z and S, Q, prior read, P,
   K, V, innovation W, raw updated S, updated Q, full read, raw/stored Z;
   forward RMS/quantiles/max and retained activation-gradient RMS/quantiles/max.
4. Encoder and inverse-decoder module calls, decoded hidden, and logits before
   and after the optimizer step.
5. Matrix-free local and composed recurrence Jacobian estimates where
   tractable.
6. Frozen-weight instantaneous comparisons on the same batch:

```text
attached / no count scaling                 historical control
horizon-detached / no scaling
attached / accumulated-read sqrt(write_count)
horizon-detached / sqrt(write_count)
```

The three counterfactuals do not update weights. They test immediate gradient
conditioning only and cannot substitute for matched retraining.

## Outputs and success criteria

- `parameter_gradients.tsv`
- `parameter_updates.tsv`
- `horizon_gradients.tsv`
- `horizon_gradient_cosines.tsv`
- `recurrent_tensor_metrics.tsv`
- `module_tensor_metrics.tsv`
- `counterfactuals.tsv`
- `jacobian_metrics.tsv`
- `interpretation.md`

Metrics are TSV and interpretation is Markdown. Temporary checkpoints and
tensors remain under ignored outputs. The audit succeeds only if:

- parameter-group squared raw gradients sum to the measured global norm;
- per-horizon gradients scaled and summed by `1/16` reproduce the total-loss
  gradient within numerical tolerance;
- the recorded parameter deltas exactly reproduce the post-step model;
- all non-finite values and conflicting directions are retained explicitly.

## Evidence boundary

This is a one-update local response audit at native step 1000. It does not
reconstruct the historical step-900 event and does not establish whether
detach or scaling prevents failure over a fresh training trajectory.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python eval_byte256_complex_self_predicted_kv_no_qkv_prenorm_step1000_one_update_forensics.py \
  --trace-examples 16 --jacobian-iterations 5
```

The registered outputs are complete. Full-batch parameter/horizon gradients
use all 64 rows; detailed tensor adjoints and shared RevNet call-role splits
use the first fixed physical microbatch of 16 rows. This scope distinction is
retained in `interpretation.md`.
