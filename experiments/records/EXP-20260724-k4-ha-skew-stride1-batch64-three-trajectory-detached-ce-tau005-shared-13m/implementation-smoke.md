# Shared-computation implementation smoke

- State: completed
- Authorization: implementation validation within the user-requested rerun
- Evidence status: excluded from the main experiment

## Question

Does sharing the encoder, clean K orbit, sigma tape, and inverse-decoder prefix
produce the same outputs and gradients as literal trajectory expansion while
reducing runtime and allocated VRAM?

## Protocol

- Unit comparison on identical inputs, weights, and Gaussian tape:
  shared versus expanded logits, latent states, CE, and gradients to K, sigma,
  and embeddings.
- One-update CUDA smoke with effective batch 64, microbatch 16, seed 1337,
  two fixed validation examples, and tau 0.05.
- The main train/validation split and checkpoints are not selected from this
  smoke.

## Success

- all equivalence tests pass within floating-point tolerance;
- one update completes without OOM;
- peak allocated VRAM and wall time are recorded before the main run.

## Result

- Ten unit tests passed, including shared-versus-literal-expansion equality
  for logits, latent tapes, and gradients to K, sigma, and token embeddings.
- Microbatch 16 completed one effective-batch-64 update.
- Final microbatch-16 peak allocated VRAM: `13,002,608,128` bytes.
- Microbatch 32 was attempted twice during memory optimization. After removing
  redundant shared-state copies, it still OOMed when exact 8192-way CE tried
  to allocate its 3 GiB softmax workspace. It is excluded from the main run.
- The registered main run therefore uses microbatch 16 with four accumulation
  passes.
