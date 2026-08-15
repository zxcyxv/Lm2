# EXP-20260729 alpha-one spectral collapse with h1-only CE

## Status

- State: preregistered before optimizer update 1
- Authorization: user requested rerunning the original end-to-end projected
  architecture with only four-token CE changed to one-token CE
- Matched parent:
  `EXP-20260728-k4-input-conditioned-spectral-collapse-alpha1-mse-ce-13m`
- Test split remains unread.

## Question

If the end-to-end hard-shell projected architecture receives token CE only
at horizon one, while every other architectural, optimization, data, and
one-step latent-loss condition remains fixed, what happens to h1 quality and
to unsupervised projected horizons two through four?

## Sole experimental change

The parent objective used:

`L_CE_parent = mean_(j=1..4) CE(Decode(u_j), gold_j)`.

This run uses:

`L_CE = CE(Decode(u_1), gold_1)`.

The complete four-state projected rollout is still constructed and jointly
decoded. Horizons two through four remain in forward execution, validation,
and checkpoints, but contribute exactly zero token-CE gradient.

Everything else is matched:

- `u_j = Pi(K_A u_(j-1))` with alpha one
- the same learned global orthogonal basis and positive amplitude shell
- the same prefix-conditioned phase and decay controller
- the same raw unprojected one-step relative MSE
- the same width-896 reversible encoder, two blocks, exact inverse decoder,
  and RMS-tied head
- the same seed, initialization order, training-window sampling, shell
  calibration, validation-start sampling, batch, microbatch, optimizer,
  schedule, report steps, checkpoint steps, and strict float32 settings

## Objective

`v_1 = K_A h_A`

`u_1 = Pi(v_1)`

`L_MSE = ||v_1 - stopgrad(h_B)||^2 / ||stopgrad(h_B)||^2`

`L = CE(Head(Decode(prefix,u_1)), gold_(A+1)) + L_MSE`.

There is no h2--h4 CE, h2--h4 latent MSE, T corrector, branch, prior, KL,
innovation, noise, or token feedback.

## Fixed configuration

- WikiText-103 train and validation; vocabulary 8192
- seed 1337; validation seed `1337 + 999`
- shell calibration seed `1337 + 9100`, 16 windows
- width 896; two reversible blocks
- alpha 1; four computed horizons; stride-one anchors 0 through 255
- CE-supervised horizons: `(1,)`
- detached raw h1 relative-MSE weight 1
- effective batch 64; microbatch 16; four accumulations
- AdamW; clip norm 1.0
- 1000 steps on the unchanged 6000-step schedule
- 128 validation examples
- test split unread

## Structural criteria

All parent preflight criteria remain required. Additionally:

- the h1-only token loss equals `mean(output.token_ce[...,0])`
- changing h2--h4 targets does not change the training token loss
- the h1-only token loss gives zero gradient to h2--h4 logits
- initial model parameters, calibration starts, validation starts, and
  calibrated shell match the parent producer under the fixed seed

## Step-1000 criteria

Primary:

- validation projected h1 NLL is below its step-1 value
- raw one-step validation relative MSE is at most `0.05`

Diagnostic:

- report projected and unprojected NLL, accuracy, and state MSE separately at
  h1 through h4
- compare all four horizons to the matched parent's step-1000 record

No h2--h4 success is claimed because those token losses are unsupervised.
Numeric metrics are TSV and interpretation is Markdown. Checkpoints and smoke
artifacts remain under `outputs/`.
