# EXP-20260724 K=3 hA clean three-token windows, 13M

## Status

- State: stopped by user after the complete step-2500 report
- Result status: incomplete; not eligible for the registered final comparison
- Authorization: user-requested
- Primary comparison:
  `../EXP-20260723-k1-ha-skew-learned-noise-mse-ce-13m/`

## Question

Does directly supervising a joint three-token `K` orbit fix the collapse at
`K^2 hA` and `K^3 hA`, and does any improvement extrapolate without
supervision to `K^4 hA` and `K^5 hA`?

## Model and exact objective

- WikiText-103 train split, BPE vocabulary 8192, seed 1337
- width 896, two reversible causal blocks, exact shared inverse decoder
- rms-tied token head
- ordinary real-token causal state `hA`; no query branch
- `K = exp(A-A^T)`, initialized as identity and exactly orthogonal
- no spectral normalization or spectral penalty
- no sigma predictor and no injected epsilon
- attached online target states; no stop-gradient is introduced

Each sampled token window has positions `0..258`. The encoder processes the
real sequence once. Anchors are `a in {0,3,6,...,255}`. For each anchor:

`z_(a,1)=K h_a`, `z_(a,2)=K^2 h_a`, `z_(a,3)=K^3 h_a`.

The state objective is the mean relative MSE

`sum_(a,j) ||z_(a,j)-h_(a+j)||^2 / ||h_(a+j)||^2`, for `j=1..3`.

For CE, `[z_(a,1),z_(a,2),z_(a,3)]` is appended as one causal future tape
behind the literal prefix ending at anchor `a`. Thus slot two sees slot one,
slot three sees slots one and two, and no slot sees a later slot. The exact
inverse decoder and tied head predict real tokens `x_(a+1..a+3)`. CE is averaged
over all 86 anchors and three horizons. Total loss is `CE + relative MSE`.

The sparse-anchor decoder is required to match independently decoded literal
prefix tapes before training.

## Run and evaluation

- 6000 updates, batch 64, 258 supervised token decisions per example
- AdamW betas 0.9/0.95, no weight decay, peak LR 3e-4, 100-step warmup and
  cosine decay, gradient clipping at 1.0
- strict float32; TF32 disabled
- fixed validation windows: seed `1337 + 999`
- fixed direct/hard rollout prefixes: seed `1337 + 7777`, 256 examples
- reports at steps 1, 50, 100, every 250 steps, and the final step
- checkpoint selection: minimum mean validation NLL across the three supervised
  horizons; test split remains unread
- report joint and per-horizon NLL/accuracy, per-horizon state MSE/cosine,
  direct-clean `K^1..K^5`, and hard-token `h1..h5`

## Registered success criteria

- Three-window success: at the NLL-selected checkpoint, the mean direct-clean
  accuracy at h2 and h3 exceeds the matched 6000-step comparison
  (`0.04297/0.03125`) by at least 0.05 absolute, while h1 falls by no more than
  0.03 from `0.37891`.
- Unsupervised extrapolation signal: mean direct-clean h4 and h5 exceeds the
  comparison (`0.03125/0.03516`) by at least 0.01 absolute.
- Any conclusion is limited to this seed, split, clean objective, and
  architecture. Training maxima do not override the checkpoint rule.

## Interruption record

The user explicitly stopped this run on 2026-07-24 to replace anchor stride
three with anchor stride one. The latest complete step-2500 report was:

- validation joint NLL `5.5637`
- validation h1/h2/h3 accuracy `0.253/0.138/0.090`
- direct-clean h1..h5 `0.246/0.141/0.086/0.051/0.043`
- hard-token h1..h5 `0.246/0.094/0.066/0.035/0.031`

The process was interrupted during the next backward pass. Existing metrics
and checkpoints remain incomplete evidence.
