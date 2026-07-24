# EXP-20260724 K=1 h_A i-ResNet spectral control, 13M

## Status

- State: stopped by user during step 4501 after the complete step-4500 report
- Result status: incomplete; not eligible for the registered final comparison
- Authorization: user-requested
- Comparison: `EXP-20260723-k1-ha-skew-learned-noise-mse-ce-13m`

## Question

Does an i-ResNet-style layerwise spectral constraint reduce the exact-inverse
decoder's sensitivity enough to improve unsupervised direct `K^2..K^5`
extrapolation, and does that constraint materially hurt one-step language
modeling?

## Matched model and objective

- WikiText-103 train split, BPE vocabulary 8192, context 256, seed 1337
- width 896, two reversible encoder blocks, exact shared inverse decoder
- no query branch; `K` acts on ordinary causal `h_A`
- `K = exp(A-A^T)`, hence `K` is already orthogonal with spectral norm one
- learned per-channel noise initialized at sigma 0.05
- unchanged objective: next-token CE plus relative
  `MSE(K h_A, h_B)`; the executed baseline's attached `h_B` gradient is retained
- rms-tied head, AdamW, peak LR 3e-4, no weight decay, strict FP32/no TF32
- 6000 updates, batch 64

No loss term, prediction layer, detach boundary, or K parameterization is
changed.

## Spectral intervention

The intervention follows Eq. (2) and the fully connected implementation from
Behrmann et al., *Invertible Residual Networks* (ICML 2019):

`W_tilde = W / max(1, sigma_tilde(W) / c)`.

- `c = 0.9`
- five direct `W`/`W^T` power iterations per optimizer update
- soft normalization: a raw weight below `c` is not expanded
- raw optimizer parameters remain unconstrained; the differentiable effective
  weight is normalized at use time
- targets: `attn.qkv`, `attn.proj`, and both FFN linear maps in each of the two
  shared reversible blocks (eight matrices total)
- embeddings, rms-tied head, learned sigma predictor, and already-orthogonal K
  are not normalized

The original i-ResNet hook updates power vectors once when a layer is called.
Here a shared layer is called repeatedly by encode and exact-inverse decode.
Power vectors are therefore updated once per top-level optimizer update and
the resulting effective weights are cached for each complete computation.
This retains Eq. (2) and the five-iteration update while ensuring both
directions use exactly the same weights. An inverse roundtrip error is logged
during training.

This is an exact implementation of the paper's **linear-weight normalization
rule**, not a claim that the complete Transformer sublayer is globally
contractive. RMSNorm, GELU, and input-dependent softmax attention violate the
paper's simple composition of non-expansive activations and constrained linear
maps. Power iteration also underestimates the true norm, as the paper notes;
exact matrix singular values are therefore audited at initialization and for
the selected/final checkpoints.

Primary sources:

- <https://proceedings.mlr.press/v97/behrmann19a.html>
- <https://github.com/jhjacobsen/invertible-resnet/blob/master/spectral_norm_fc.py>

## Split, selection, and evaluation

- fixed validation starts: seed `1337 + 999`
- fixed extrapolation starts: seed `1337 + 7777`, 256 prefixes
- checkpoint selection: minimum fixed-validation next-token NLL among scheduled
  reports; test split remains unread
- reports at steps 1, 50, 100, every 250 steps, and the final step
- language metrics: validation NLL and top-1 accuracy
- extrapolation metrics:
  - direct-clean `K^1..K^5` with no token re-encoding
  - hard-token rollout with re-encoding after each greedy action
- mechanism audits: relative state MSE/cosine, learned log-sigma, K
  orthogonality, estimated/exact spectral norms, analytic inverse roundtrip

## Registered success criteria

- Extrapolation improvement: at the NLL-selected checkpoint, mean direct-clean
  accuracy over `h2..h5` exceeds the matched baseline's selected checkpoint by
  at least 0.01 absolute while direct `h1` falls by no more than 0.02.
- Material language harm: selected validation NLL is at least 0.05 worse and
  validation accuracy at least 0.01 lower than the matched baseline.
- Any conclusion is limited to this seed, split, coefficient, and architecture.
  Training-curve maxima do not override the checkpoint rule.

## Interruption record

The user explicitly stopped this run on 2026-07-24 to replace it with a clean
three-token-window experiment. The latest complete report is step 4500:
validation NLL 4.0209, accuracy 0.2971, direct-clean h1..h5
`0.309/0.039/0.039/0.023/0.043`. The process was interrupted during the next
backward pass. Existing logs and checkpoints are retained as incomplete
evidence.
