# EXP-20260802 unitary branch-normalized H16 continuation to 6000, micro64

## Status

- State: completed
- Parent: step-1000 exact continuation record
- Resume source: parent `step1000.pt`
- Test split remains unread.

## Question and intervention

What are the step-6000 stability, NLL, and real-sample interpretability of the
unchanged branch-normalized unitary H16 model? The sole execution intervention
is regrouping the same effective batch 64 from microbatch 16 x 4 to microbatch
64 x 1 to use the available 20GB GPU and reduce runtime.

The batch samples, mean CE objective, optimizer state, data-generator state,
LR schedule and clipping remain unchanged. Floating-point summation order and
therefore the exact numerical trajectory may differ after step 1000.

The run completed at step 6000 with finite metrics. Post-run benchmark and
audits are registered as separate records so this training record remains an
unchanged producer result.

## Fixed protocol

- Resume model/AdamW/data generator at step 1000
- effective batch 64; microbatch 64; no accumulation
- H16 stride 16; fully attached CE
- peak LR `3e-4`; original schedule ends at step 6000; clip 1.0
- report 1001 and every 500 steps through 6000
- validation 64 examples, evaluation microbatch 2
- first step must fit within available 20GB and remain finite

## Success and follow-up

All gradients and metrics must remain finite with no gnorm above 10 after
step 1000. Final block NLL and H1--H16 NLL are compared with step 1000. After
training, rerun the four scaling/Jacobian assumptions and the real-context
interference/decode audit at step 6000.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_stride16_6000step_micro64.py \
  --steps 6000 --batch 64 --microbatch 64 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2 \
  --resume-checkpoint outputs/experiments/EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-normalized-residual-h16-ce-only-attached-stride16-1000step-13m/step1000.pt \
  --allow-resume-microbatch-change \
  --record-dir experiments/records/EXP-20260802-byte256-unitary-branch-normalized-h16-6000step-micro64 \
  --output-dir outputs/experiments/EXP-20260802-byte256-unitary-branch-normalized-h16-6000step-micro64
```
