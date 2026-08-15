# EXP-20260729 K=1 CE with attached self-canonical T

## Status

- State: preregistered before optimizer update 1
- Authorization: user requested implementation, training, and a subsequent
  three-policy generation evaluation
- Parent architecture:
  `EXP-20260728-k4-ce-only-spectral-detached-t-canonicalization-13m`
- Test split remains unread.

## Question

Can a state-conditioned spectral operator `T(q)` learn to map the raw
one-token proposal `q = K_A h_A` toward the canonical encoder state obtained
by reinserting q's own selected token, when the regression gradient is
allowed to update q, K, the shared basis, and the encoder, while token CE
continues to decode q without T?

## Architecture

The backbone remains width 896 with two reversible causal encoder blocks,
the analytic exact-inverse decoder, and the RMS-tied embedding head.

For every real causal state `h_A`, the existing prefix-conditioned normal
spectral operator produces one proposal:

`q_A = K_A h_A`.

Only q is passed to the inverse decoder for ordinary next-token CE:

`logits_A = Head(Decode(prefix_A, q_A))`.

The selected action is target-independent:

`b_A = argmax(logits_A)`.

The literal prefix and b are passed through the same encoder to form:

`hstar_A = Encode(prefix_A, b_A)`.

The argmax and canonical target are non-differentiable. `hstar_A` is a fixed
regression target. It is not the corpus next-token state.

T is an identity-initialized width-128 MLP controller. Given q, it emits 448
bounded log-amplitude corrections and 448 bounded phase corrections in the
same learned spectral basis as K. For fixed q this represents a
complex-diagonal operator; the complete map is nonlinear:

`qhat_A = T(q_A) q_A`.

Unlike the parent experiment, q and the shared basis are not detached in the
T path. Consequently T regression gradients reach T, q, K, the shared basis,
and the encoder. They do not reach the fixed canonical target.

The CE decoder always receives raw q, never qhat. T therefore cannot improve
the current token by acting as a readout bypass.

## Objective

`L_CE = CE(logits_A, gold_(A+1))`

`L_T = ||qhat_A - stopgrad(hstar_A)||^2 / ||stopgrad(hstar_A)||^2`

`L = L_CE + L_T`.

There is no four-horizon CE, corpus-gold latent target, KL, branch,
responsibility, innovation, corrected-state token CE, or T output in the
decoder path.

## Fixed configuration

- WikiText-103 train and validation; BPE vocabulary 8192
- seed 1337; validation starts use seed `1337 + 999`
- test split unread
- width 896; two reversible causal blocks
- exact inverse causal decoder and RMS-tied head
- prefix-conditioned normal spectral K with alpha zero
- K controller bottleneck 128
- attached T controller bottleneck 128
- T maximum absolute log-scale 2.0
- T maximum absolute phase correction pi
- T identity initialization
- one-token horizon; stride-one anchors 0 through 255
- effective batch 64; physical microbatch 16; four accumulations
- one AdamW optimizer; clip norm 1.0
- corrector weight 1.0
- 1000 updates on the unchanged 6000-update LR schedule
- validation examples 128
- reports at steps 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at steps 100, 250, 500, 750, and 1000
- strict float32; TF32 disabled

The amplitude shell is deterministically calibrated from 16 training windows
using seed `1337 + 9100`, then frozen. It has no training-forward effect and
is preserved solely for the preregistered post-training collapse-projection
evaluation.

## Structural criteria

- T is identity at initialization below relative error `1e-6`
- raw q and logits are independent of T parameters
- changing the corpus future token does not change q, selected b, hstar, or
  qhat
- batched self-action encoding matches literal prefix-plus-b encoding below
  `5e-5`
- CE gives zero or absent gradient to T
- T regression gives finite nonzero gradients to T, q/K controller, shared
  basis, and encoder
- T regression gives zero or absent gradient to the canonical target
- raw CE logits are decoded from q rather than qhat
- all states, logits, losses, and gradients are finite

## Step-1000 criteria

Primary:

- validation one-token NLL is below its step-1 value
- corrected canonical-state relative MSE is below paired raw-q MSE
- corrected/raw canonical MSE ratio is at most 0.75

Supporting:

- all T scale and phase values remain finite
- raw next-token accuracy remains nonzero
- corrected-state decoding preserves q's selected token at least 90% of the
  time

Numeric metrics are TSV and interpretation is Markdown. Checkpoints and smoke
artifacts remain under `outputs/` and are not added to Git.
