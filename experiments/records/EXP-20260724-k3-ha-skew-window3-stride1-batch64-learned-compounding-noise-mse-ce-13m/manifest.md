# EXP-20260724 K=3 learned compounding trajectory noise

## Status

- State: completed at step 100
- Authorization: user-requested
- Primary comparison:
  `../EXP-20260724-k3-ha-skew-clean-window3-stride1-batch64-mse-ce-13m/`

## Question

Can learned reparameterized noise become a branch-carrying trajectory
perturbation when propagated noise participates in the later hidden-state MSE,
rather than appearing only at a one-step decoder input?

## Registered forward and loss

For clean powers `cj=K^j hA`, a shared SigmaPredictor receives each `cj`.
Independent standard-normal draws are reparameterized as
`ej=exp(log_sigma(cj))*rms(cj)*xi_j`.

The attached-target MSE states are

- `s1 = K hA`;
- `s2 = K(s1+e1) = K^2 hA + K e1`;
- `s3 = K(s2+e2) = K^3 hA + K^2 e1 + K e2`.

They regress to attached `hB/hC/hD`. The joint causal inverse decoder receives
`[s1+e1,s2+e2,s3+e3]` for the three CE terms. Thus each local noise affects
the current token readout and, except at the last training horizon, the later
MSE trajectory.

All gradients remain attached. The noise predictor input is the clean power,
not the propagated noisy state.

## Fixed configuration

- WikiText-103 train/validation, vocabulary 8192, seed 1337
- batch 64, context 256, anchor stride 1, horizons 3
- width 896 and two reversible causal blocks
- exact shared inverse decoder, rms-tied head, `K=exp(A-A^T)`
- shared per-channel SigmaPredictor, initial sigma 0.05
- independent Gaussian draw at each horizon
- no spectral constraint or any stop-gradient
- mean three-horizon CE plus mean attached-target relative MSE
- AdamW, 100-step linear warmup, gradient clip 1.0

## Split and evaluation

- run length: 100 updates
- fixed validation starts: seed `1337+999`, 128 examples
- fixed validation noise tape: seed `1337+5151`
- fixed h1..h5 rollout starts: seed `1337+7777`, 256 examples
- fixed rollout noise tape: seed `1337+6161`
- reports at steps 1, 50, and 100
- test split remains unread

## Registered success criteria

At step 100, relative to the clean attached-target control:

- mean noisy validation h2/h3 accuracy improves by at least 0.005 over
  `0.054291`;
- h1 accuracy falls by no more than 0.01 from `0.092529`;
- mean validation NLL is no higher than `7.122284`;
- learned log-sigma and its horizonwise MSE gradients are recorded so a
  collapsed-noise result is not interpreted as support.

Any result is limited to early optimization and this seed.

## Result

The registered improvement was not observed at step 100.

| Metric | Learned compounding noise | Clean control | Difference |
| --- | ---: | ---: | ---: |
| validation joint NLL | 7.135864 | 7.122284 | +0.013580 |
| h1 accuracy | 0.091125 | 0.092529 | -0.001404 |
| h2 accuracy | 0.055237 | 0.057556 | -0.002319 |
| h3 accuracy | 0.046631 | 0.051025 | -0.004395 |
| mean h2/h3 accuracy | 0.050934 | 0.054291 | -0.003357 |
| h1 relative MSE | 0.012653 | 0.012155 | +0.000498 |
| h2 relative MSE | 0.023977 | 0.022225 | +0.001752 |
| h3 relative MSE | 0.041306 | 0.037473 | +0.003833 |

The h1 tolerance criterion passed, but the h2/h3 improvement and joint-NLL
criteria failed. On the fixed noisy rollout tape, direct h1 accuracy was
`0.109375` versus the clean control's `0.101562`, while direct h2/h3 were
lower (`0.054688/0.035156` versus `0.066406/0.039062`). Hard-token h1 was
also `0.109375`, but hard-token h2/h3 were lower
(`0.039062/0.027344` versus `0.042969/0.039062`).

Mean learned sigma fell from `0.05` to approximately `0.010227`
(`mean(log_sigma)=-4.582743`). Thus propagation through later MSE terms
prevented neither substantial noise shrinkage nor the h2/h3 degradation.
Within this early one-seed comparison, the proposed branch-carrying benefit is
not supported.

- Metrics: [`metrics.tsv`](metrics.tsv)
- Run metadata and preflight gradients: [`run.tsv`](run.tsv)
- Checkpoint:
  `../../../outputs/experiments/EXP-20260724-k3-ha-skew-window3-stride1-batch64-learned-compounding-noise-mse-ce-13m/last.pt`
- Checkpoint SHA-256:
  `92143649de328935799283bd591cdcbb339d0d8d054bd3bcff8158e73668c695`
