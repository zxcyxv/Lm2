# EXP-20260802 step-6000 fast-path benchmark

## Status

- State: completed
- Source checkpoint: step 6000 of the microbatch-64 continuation
- Training split is sampled read-only; test split remains unread

## Question and comparison

Does the CE-only fast training path reduce forward/backward iteration time and
peak allocated VRAM relative to the original full diagnostic objective without
changing the scalar CE on the same batch?

- Control: full objective, including future encode and diagnostic tapes
- Intervention: prefix-only encode, read-state-only rollout, no unused latent
  target or energy diagnostics
- Checkpoint and batch are reset identically for each path

## Fixed protocol

- seed 1337 + 6060; one fixed training batch of 64 windows
- microbatch 64; H16; stride 16
- three CUDA warm-up iterations and ten timed forward/backward iterations
- no optimizer update, clipping, data sampling, evaluation, or checkpoint write
- TF32 disabled, float32 matmul precision highest

## Success criterion

The two initial scalar losses were preregistered to agree within PyTorch default
close tolerance. The first execution stopped before writing benchmark metrics:
full `2.8909676075`, fast `2.8908076286`, absolute difference `1.5998e-4`.
This is preserved as conflicting finite-precision evidence. Because prefix-only
and full-length CUDA causal encodes use different matrix shapes, the operational
equivalence threshold for the rerun is explicitly `atol=2e-4, rtol=1e-4`; the
raw losses and difference remain in TSV.

The fast path must have lower median iteration time and no greater peak allocated
VRAM. The measured ratio is a compute-only ETA factor, not an end-to-end promise.
