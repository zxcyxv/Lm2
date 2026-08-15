# EXP-20260727 K=1 exact-inverse CE-only solver replication

## Status

- State: preregistered before optimizer update 1
- Authorization: user requested an approximately 1000-step training run
- Test split remains unread

## Question

Can the minimal `encoder -> K -> exact-inverse decoder` model be trained with
ordinary next-token teacher forcing alone, providing a checkpoint on which to
measure whether parallel latent initialization plus inference-time causal
solving recovers the model's own greedy AR trajectory?

This training run does not test the solver.  Solver iteration counts and AR
agreement will be measured in a separately recorded inference audit.

## Fixed model and objective

- vocabulary 8192, width 896, two reversible causal encoder blocks
- one shared bias-free linear latent operator `K`
- analytic exact-inverse decoder
- RMS-normalized tied vocabulary head
- loss: next-token cross entropy at every teacher-forced position
- no latent MSE, closure loss, branch, trajectory loss, noise, KL, or
  distillation

This is a replication of the `inverse_rms` arm of
`EXP-20260721-k1-decoder-inverse-ablation-13m`, isolated under a new record so
the historical evidence is not overwritten.

## Data, seed, split, and optimization

- local WikiText-103 BPE train and validation splits
- seed 1337 for initialization, sampled training windows, and fixed validation
  starts (`seed + 999`)
- context 256
- batch 128
- 1000 optimizer updates
- AdamW, peak learning rate `3e-4`, 100-step warmup, cosine decay, gradient
  clipping at 1.0
- float32 with TF32 matrix multiplication enabled
- 128 fixed validation windows; test split remains unread

## Controls

- Historical control: the original `inverse_rms` run under
  `EXP-20260721-k1-decoder-inverse-ablation-13m`
- The architecture, seed, split, schedule, and producer are held identical;
  only the record/output location and selected single variant differ.

## Success criteria

- finite logits and a finite nonzero gradient to `K` in preflight
- all 1000 updates complete without numerical failure
- validation NLL at step 1000 is lower than at step 1
- the exact-inverse round-trip relative error remains below `1e-4`

Passing these criteria establishes a usable CE-only checkpoint.  It does not
establish parallel generation equivalence; that is the next audit.
