# EXP-20260724 K=3 hA clean window-3, stride 1, batch 64

## Status

- State: stopped at the last complete report/checkpoint, step 750
- Authorization: user-requested correction
- Stop reason: superseded by the user-requested truncated-gradient run
- Primary comparison:
  `../EXP-20260724-k3-ha-skew-clean-window3-mse-ce-13m/`

The preserved `last.pt` is step 750 with SHA-256
`b9709a7a490c09359f0122c9205fee4da30f29287dbe19bb0a53e54af11162e7`.
At that checkpoint, validation h1/h2/h3 accuracy was
`0.200623/0.114868/0.080597`; this is the matched-step control for the
truncated-gradient run.

## Question and sole configuration change

Does changing anchors from `0,3,6,...,255` to every position
`0,1,2,...,255` improve the clean three-token trajectory when the original
batch size 64 is retained?

All other configuration is unchanged:

- WikiText-103 train/validation, vocabulary 8192, seed 1337
- batch 64, width 896, two reversible causal blocks
- exact shared inverse decoder and rms-tied head
- `K=exp(A-A^T)`, initialized identity and exactly orthogonal
- clean joint `[K hA,K^2 hA,K^3 hA]` causal tape
- attached `[hB,hC,hD]` relative MSE plus three-position CE
- no spectral constraint, sigma predictor, noise, or stop-gradient
- 6000 updates, identical AdamW and LR schedule

The denser anchors increase supervised labels per update from 16,512 to
49,152. This is intentionally part of the user-requested stride intervention;
the run is update-matched, not label-count-matched.

## Split, selection, and evaluation

- fixed validation starts: seed `1337+999`
- fixed h1..h5 rollout starts: seed `1337+7777`, 256 examples
- reports at 1, 50, 100, every 250 steps, and final
- checkpoint selection: minimum mean validation h1..h3 NLL
- test split remains unread

## Registered success criteria

At step 2500, compared with the stopped stride-3/batch-64 run:

- mean validation h2/h3 accuracy improves by at least 0.01 from 0.114;
- h1 accuracy falls by no more than 0.02 from 0.253;
- direct-clean mean h2/h3 improves by at least 0.01 from 0.1135.

Any result is limited to this seed and the fact that denser anchors also triple
the supervised labels per optimizer update.
