# EXP-20260801 online stop-gradient H1 complex KV regression

## Status

- State: stopped by user after the step-250 report; superseded by the EMA
  target/inference experiment
- Matched parent:
  `EXP-20260801-byte256-complex-self-predicted-kv-full-innovation-h1-online-mse-ce-13m`
- Test split remains unmaterialized and unread.

## Question

Does stopping only the gradient into the same online encoder's H1 future-state
target reduce coordinate chasing and improve self-composed equivalence to the
same model's greedy AR path?

This is a one-factor backward-path ablation of the parent. Forward values,
model initialization, data order, recurrence, CE, relative-MSE value, and
evaluation are identical at step 0.

## Fixed model and causal recurrence

- byte vocabulary 256; width 1344; two reversible causal encoder blocks
- analytic exact-inverse decoder
- fixed regular-simplex input/output codebook and raw tied logits
- 8 complex-memory heads; complex key dimension 16; value dimension 31
- learned unit-modulus hidden and memory rotations; no decay
- self-predicted coefficient-free rank-one innovation
- no beta, temperature, gate, residual prior, independent vocabulary head,
  EMA, sampled branch, or observed/generated-token KV write

For one transition:

`h_prior = Read(q(h_rot), U S)`

`I = k(h_prior) v(h_prior)^dagger`

`h_post = h_prior + Read(q(h_prior), I)`

`S_next = U S + I`

Only `h_prior` supplies the current-token CE readout. `h_post` is the next
recurrent hidden and the H1 latent prediction.

## Only changed line

Parent:

`L_H1 = ||h_post - h_B_online||^2 / ||h_B_online||^2`

This experiment:

`L_H1 = ||h_post - sg(h_B_online)||^2 / ||sg(h_B_online)||^2`

Total loss remains:

`L = CE(raw_tied_logits(Decode(h_prior)), B) + L_H1`

The future state is produced by the same current online encoder pass. It is
not an EMA or separate model. There is no KL divergence, decoded-state loss,
logit matching, or gradient into `h_B_online`. The future token/state appears
only on the loss side and never enters the central recurrence.

## Data and fixed execution

- WikiText-103 raw UTF-8 train and validation bytes only
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed `1337 + 999`
- context 256; training horizon 1; stride-one anchors
- monitoring horizon 4; boundary stride 4
- effective batch 64; physical microbatch 16
- strict float32; TF32 disabled
- AdamW, peak LR `3e-4`, clip norm 1.0
- 1000 updates on the fixed 6000-update schedule
- reports at 0, 1, 50, 100, 250, 500, 750, and 1000

## Structural preflight

- forward loss, prior/posterior states, and logits equal the attached-target
  formulation before backward
- H1 target gradient is absent exactly
- CE and H1 regression retain finite nonzero gradients to the online prefix
  encoder, complex projections, phases, and innovation readout
- held-out future-byte mutations do not affect the central tape or main logits
- H1 parallel and greedy-AR paths agree before feedback
- no beta parameter exists

## Step-1000 success criteria

All are required:

- H1 validation NLL improves from step 0
- H1 online-SG relative MSE improves from step 0
- H1 posterior target accuracy exceeds the parent's `0.109619`
- H2--H4 central/greedy-AR agreement exceeds the parent's `0.261149`
- write-enabled H2--H4 agreement exceeds functional no-write agreement
- four-horizon mean central/greedy-AR state cosine is at least `0.75`
- H1 central/AR max logit error remains below `5e-4`

Metrics are TSV and interpretation is Markdown. Checkpoints remain ignored.

## Stopped-run boundary

The run was interrupted before the step-500 report. `metrics.tsv` therefore
contains complete reports only through step 250, and the retained checkpoints
are `step0100.pt`, `step0250.pt`, and `last.pt`. This partial evidence is not a
step-1000 outcome and is not promoted to the preregistered success decision.
