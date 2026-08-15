# EXP-20260803 time-varying scan fusion benchmark

## Status

- State: fused rotating-frame follow-up and direct trainer timing complete
- Parent: completed RTX-5090 time-varying scan step-1000 record
- Training split only for fixed benchmark windows; test unread

## Question and comparison

Can the eager Hillis--Steele affine scan, which achieved only `98026` nominal
bytes/s, be replaced by a generated CUDA/Triton associative-scan kernel that
materially reduces full forward/backward time? Compare against the preserved
eager scan and the preserved feedback-trainer speed; do not rerun either
training baseline.

Before finalizing the kernel path, also test the more specific unitary
integrating-factor form. Because both homogeneous transports are invertible
channelwise rotations, all prefix states can be written exactly as a
rotating-frame cumulative sum. This candidate is preferred if it matches the
literal recurrence and is faster than the generic associative scan.

Follow-up: fuse cumulative phase construction, inverse rotation, inclusive
addition scan, and forward rotation into one Triton kernel for each carrier.
Compare this fused rotating-frame backend with both the unfused rotating-frame
path and the existing fused affine Triton path under the same fixed protocol.
The backward must likewise compute the reverse rotating-frame suffix scan,
increment/root adjoints, and phase contribution in one Triton kernel.

## Fixed protocol

- RTX 5090, strict float32, TF32 disabled
- scan step-1000 checkpoint and one seed-fixed training batch of 64
- H16, stride 16, identical loss and decoder
- warm up generated kernels before timed repeats
- first test fused forward algebra; add a custom backward only if forward
  timing indicates a plausible end-to-end gain
- compare the eager generic scan, generated associative scan, Triton scan, and
  rotating-frame `cumsum` on the same checkpoint and batch
- after the controlled benchmark, run 200 optimizer steps from the fixed
  initialization with batch 64, seed 1337, the ordinary 6000-step learning
  rate schedule, and the 64 fixed validation examples; use training-wall time only for
  the direct ETA confirmation
- repeat the same 200-step timing with the fused Triton rotating-frame backend;
  success requires finite training and a steady segment faster than the
  unfused rotating-frame trainer

## Success and implementation gate

Forward state values must agree with the eager scan within normal float32
reassociation tolerance. A training implementation is warranted only if the
candidate removes at least 15% of scan-path time or profiling shows an
equivalent end-to-end opportunity. Any custom backward must have finite
gradients and agree in direction and norm with eager autograd before it can be
used for training. Metrics go to TSV and conclusions to Markdown.
