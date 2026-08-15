# EXP-20260803 unitary time-varying H16 scan, RTX 5090

## Status

- State: registered; execution pending
- New scan arm only; the feedback baseline will not be rerun
- Training and validation splits only; test remains unread

## Question and comparison

Can a time-varying central recurrence preserve exact recurrent prefix
composition while replacing the H16 state-dependent execution chain with
associative scans? Measure its RTX 5090 speed and early training performance.

The quality reference is the existing feedback-recurrence step-1000 record:
H1 NLL `2.049449`, block NLL `3.054808`, and H16 NLL `3.148452`. The hardware
speed reference is the existing RTX-5090 200-step feedback record:
`0.120026 s/step`, `144376 nominal sampled bytes/s`, and `7.254 GiB` peak VRAM.

## Scan intervention

For each initial encoder root, first compile the time-varying unitary orbit
`c_h = R^h z_0` and all `q_h,k_h,v_h` in parallel. Then compute the exact
causal prefixes of

```text
S_h = U S_(h-1) + k_h v_h^dagger
delta_h = O(q_h^dagger S_h / ||S_h||_F)
z_h = R z_(h-1) + delta_h
```

with a complex-memory affine scan followed by a pairwise-orthogonal latent
affine scan. Thus every returned `z_h` is the literal recurrent successor and
the encoder/central/inverse-decoder composition remains a recurrence. The sole
mechanistic removal is measurement-updated `z_h -> future QKV`; future QKV is
the precompiled time-varying forcing tape instead.

## Fixed protocol

- seed 1337; same model initialization and training-window RNG as baseline
- WikiText-103 byte train split and the baseline's 64 fixed validation starts
- H16, stride 16, effective batch = microbatch = 64
- 1000 optimizer steps under the unchanged 6000-step LR schedule
- AdamW `(0.9,0.95)`, no weight decay, global raw-gradient clip 1.0
- strict float32, TF32 disabled, one RTX 5090
- validation at steps 0, 100, 300, and 1000

## Success and scope

The generic memory scan and complete central scan must match literal sequential
evaluation within test tolerance, and H1 feedback/scan logits must differ by
at most `5e-5`. Training succeeds if all losses and gradients remain finite.
Speed is reported rather than required to improve. Step-1000 horizon NLLs are
compared with the preserved baseline record; this is an early trainability
comparison, not a final quality claim.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_unitary_time_varying_scan_h16_1000.py
```
