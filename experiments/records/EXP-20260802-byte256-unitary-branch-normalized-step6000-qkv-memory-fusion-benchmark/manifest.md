# EXP-20260802 step-6000 QKV and memory-fusion benchmark

## Status

- State: registered; execution pending
- Source checkpoint: completed step-6000 branch-normalized H16 model
- One fixed training batch; test unread

## Question and comparison

Can checkpoint-compatible single-GEMM QKV and a real-pair compiled memory
rotation/write/Frobenius/read region accelerate the current CE-only fast path
without changing its mechanism?

- Preserved control: prior fast-path median 486.430 ms, peak 7.892 GB
- Candidate: pack existing Q/K/V weights once per rollout, project with one
  GEMM per recurrent step, and execute the complex memory region as equivalent
  real/imaginary algebra in a full-graph compiled CUDA region

## Fixed protocol

- same checkpoint, seed 1337 + 6060, batch=microbatch 64, H16 stride 16
- three warm-up and ten measured forward/backward iterations
- full diagnostic path supplies a same-run numerical reference; compiled
  memory fusion is active only for the diagnostics-free CE training rollout
- matched QKV-only ablation disables the compiled memory region and writes to
  the `qkv_only/` subrecord, separating the two contributions
- no optimizer update, data sampling, evaluation, or checkpoint mutation

After controlled timing, a fresh 100-step actual-trainer subrun uses batch and
microbatch 64, schedule 6000, and one validation example at the endpoints. It
includes first compilation, optimizer, clipping, sampling and checkpointing and
writes to `trainer_eta100/` plus an ignored output directory.

The first trainer attempt stopped before optimization because the inherited
preflight required bitwise-zero agreement between separately recomputed Q/K/V
GEMMs and the packed single GEMM. Measured maximum discrepancies were
`1.23e-6` for memory, `1.12e-8` for the measurement delta, and `1.19e-7` for
the successor. The candidate preflight therefore uses an explicit `2e-6`
maximum absolute formula tolerance; the stopped attempt remains part of scope.

A final QKV-only 100-step trainer subrun disables the candidate memory kernel
and records the configuration selected for future training in
`qkv_only_eta100/`.

## Success criteria

- candidate loss differs from the same-run diagnostic reference by at most
  `atol=2e-4, rtol=1e-4`, preserving the earlier finite-precision boundary
- median candidate iteration time must beat 486.430 ms
- all backward gradients finite; peak allocation reported separately

Before acceptance, a matched gradient audit uses the same fast objective and
step-6000 checkpoint with only the compiled memory flag toggled. It records
loss difference, global gradient cosine and relative L2 error on a fixed
two-window training batch in `gradient_equivalence.tsv`.
