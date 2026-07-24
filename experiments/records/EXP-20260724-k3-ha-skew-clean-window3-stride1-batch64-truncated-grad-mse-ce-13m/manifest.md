# EXP-20260724 K=3 stride-1 truncated horizon gradient

## Status

- State: stopped at step 100
- Authorization: user-requested
- Stop reason: severe matched-step regression; user requested theoretical review
- Primary comparison:
  `../EXP-20260724-k3-ha-skew-clean-window3-stride1-batch64-mse-ce-13m/`

The preserved `last.pt` is step 100 with SHA-256
`a23a9f1fe20e23d522b2edad971dafcc6080cb145a9a9dca99300e336d9d12b1`.
At step 100, validation h1/h2/h3 accuracy was
`0.035095/0.035004/0.034851`, versus
`0.092529/0.057556/0.051025` without truncation. This run fails the intended
early protection signal and is not continued to the registered step-750
criterion.

## Question and sole gradient intervention

Does truncating backward credit between predicted horizon states protect h1
learning while retaining useful h2/h3 rollout learning?

Forward values and losses are unchanged. Let
`z1=K(hA)`, `z2=K(sg(z1))`, and `z3=K(sg(z2))`. For a later inverse-decoder
slot, earlier orbit slots have identical forward key/value numbers but are
stop-gradient views. Thus later CE cannot update earlier orbit state tensors
through causal branch attention.

The shared K, decoder, head, and encoder parameters remain trainable at every
horizon. Gold `hB/hC/hD` targets remain attached; target stop-gradient is
explicitly deferred to a later experiment.

All other configuration is retained:

- WikiText-103 train/validation, vocabulary 8192, seed 1337
- batch 64, context 256, anchor stride 1, horizons 3
- width 896 and two reversible causal blocks
- exact shared inverse decoder, rms-tied head, `K=exp(A-A^T)`
- no noise, sigma predictor, or spectral constraint
- mean h1/h2/h3 CE plus mean attached-target relative MSE
- AdamW, the existing learning-rate schedule, gradient clip 1.0, 6000 updates

## Split, selection, and evaluation

- fixed validation starts: seed `1337+999`, 128 examples
- fixed h1..h5 rollout starts: seed `1337+7777`, 256 examples
- reports at 1, 50, 100, every 250 steps, and final
- checkpoint selection: minimum mean validation h1..h3 NLL
- test split remains unread

## Registered success criteria

At the matched step 750, relative to the non-truncated control:

- validation h1 accuracy improves by at least 0.01 over `0.200623`;
- mean validation h2/h3 accuracy falls by no more than 0.01 from `0.097733`;
- h1 relative MSE does not increase by more than 10% from `0.010948`.

This tests truncated backward credit only. It does not test target
stop-gradient or a different horizon-loss weighting.
