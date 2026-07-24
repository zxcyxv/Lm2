# EXP-20260724 K=3 hA clean window-3, anchor stride 1

## Status

- State: stopped by user after step 250
- Result status: excluded from the requested stride comparison because the
  agent changed batch size from 64 to 22
- Authorization: user-requested
- Primary comparison:
  `../EXP-20260724-k3-ha-skew-clean-window3-mse-ce-13m/`

## Question

Does supervising a three-token `K` orbit at every token position, rather than
only every third position, improve the direct h2/h3 trajectory and downstream
generation quality?

## Only intended intervention

- comparison anchors: `0,3,6,...,255`
- this run's anchors: `0,1,2,...,255`

Everything else remains the clean three-token architecture:

- WikiText-103 train/validation, BPE vocabulary 8192, seed 1337
- width 896, two reversible causal blocks, exact shared inverse decoder
- rms-tied token head
- `K = exp(A-A^T)`, initialized as identity and exactly orthogonal
- no spectral constraint, sigma predictor, injected noise, or stop-gradient
- joint causal branch `[K hA,K^2 hA,K^3 hA]`
- relative MSE against `[hB,hC,hD]` plus CE for the three decoded real tokens

## Label-count matching and run

- 256 anchors × 3 horizons = 768 labels per example
- batch 22 = 16,896 supervised labels per update
- stride-3 comparison: 86 anchors × 3 horizons × batch 64 = 16,512 labels per
  update
- this keeps label/update count within 2.3%, while the number of independently
  sampled windows per update changes from 64 to 22 and is a stated confound
- 6000 updates; AdamW 0.9/0.95, no weight decay, peak LR 3e-4, 100-step warmup,
  cosine decay, gradient clipping 1.0
- strict float32 with TF32 disabled

## Split, selection, and evaluation

- fixed validation windows: seed `1337 + 999`
- fixed rollout prefixes: seed `1337 + 7777`, 256 examples
- reports at 1, 50, 100, every 250 steps, and final
- checkpoint selection: minimum mean validation NLL over h1..h3
- report horizon NLL/accuracy/state geometry and direct/hard h1..h5
- test split remains unread

## Registered success criteria

At step 2500, relative to the stopped stride-3 run:

- mean validation h2/h3 accuracy improves by at least 0.01 absolute from
  `(0.138+0.090)/2 = 0.114`;
- h1 accuracy falls by no more than 0.02 from `0.253`;
- direct-clean mean h2/h3 improves by at least 0.01 from
  `(0.141+0.086)/2 = 0.1135`.

Final language and generation quality will be judged at the NLL-selected
checkpoint; early maxima do not override checkpoint selection. Conclusions
remain limited to this seed and the stated batch/window confound.

## Interruption record

This run used an agent-selected batch-size change that the user did not intend.
It was stopped and replaced with a batch-64 run. Its metrics remain provenance
only and must not be used as the stride-1 result.
