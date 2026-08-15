# EXP-20260801 P-detached joint hidden and decoder closure

## Status

- State: preregistered; not yet executed
- KL-only control:
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-self-redecode-kl-ce-13m/`
- EMA hidden-MSE control:
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-sg-mse-ce-13m/`
- Test split remains unmaterialized and unread.

## Question

Can a shared self-selected EMA target make Znext match both the canonical
hidden state and its decoded token distribution, while stopping successor
closure at the CE-visible P boundary reduces CE interference?

## Fixed forward recurrence

Forward values are unchanged from both controls:

```text
P       = O(Read(Q(Zrot), Mrot))
W       = K(P) outer conj(V(P))
Mnext   = Mrot + W
Znext   = O(Read(Q(P), Mnext))
```

- P alone enters the online exact-inverse decoder for ordinary corpus CE.
- Znext is the recurrent hidden.
- No corpus-future, observed, selected, or generated token enters the central
  recurrence or central memory.
- There is no beta, temperature, gate, decay, residual prior, L2
  normalization, or independent vocabulary head.

## Backward boundary

Only the backward graph differs:

```text
P_ce       = P                         # attached to ordinary CE
P_correct  = stop_gradient(P)          # same forward value
W          = K(P_correct) outer conj(V(P_correct))
Znext      = O(Read(Q(P_correct), Mrot + W))
```

Both closure terms use this same P-detached successor graph. Detaching KL but
not hidden MSE would leave a second closure path into P and is therefore not
the registered comparison.

The pre-update matrix memory remains attached. It is part of the dynamical
state and must learn a coordinate system capable of supporting canonical
successors. The encoder and shared transition parameters are not globally
frozen. Consequently this experiment removes the direct corrector-to-P edge;
it does not claim to eliminate every possible shared-parameter gradient
conflict.

## Self-selected EMA target and objective

For every H1 anchor:

```text
online P logits -> greedy token y
EMA Encode(prefix,y) -> canonical hidden H(y)
EMA inverse-decode H(y) -> canonical log-softmax R
EMA inverse-decode online Znext -> student log-softmax S

loss = CE(online Decode(P), observed B)
     + relative_MSE(Znext, H(y))
     + forward_KL(R || S)
```

- CE weight: 1.0
- canonical hidden relative-MSE weight: 1.0
- canonical forward-KL weight: 1.0
- native softmax temperature: 1.0
- training horizon: H1 only
- y is P's detached greedy prediction, not the corpus-future byte
- the canonical EMA branch is gradient-free
- EMA parameters are frozen while the EMA inverse remains differentiable
  with respect to online Znext

The hidden and distribution targets describe the same self-selected token.
The experiment therefore does not mix corpus H(B) with a different
self-selected y target.

## Controls at step 1000

KL-only control:

- H1 validation NLL: 2.347617
- H1 Znext/canonical relative MSE: 0.996537
- H1 Znext/canonical cosine: 0.061611
- H1 canonical-to-Znext KL: 0.715272
- H1 P/Znext top-1 agreement: 0.817871
- H1 Znext mean max probability: 0.571202
- H1 Znext 99%-confidence fraction: 0.036865

EMA hidden-MSE control used the observed corpus-future hidden rather than the
self-selected target and is therefore mechanistic context, not an identical
target control:

- H1 validation NLL: 1.394581
- H1 latent relative MSE: 0.335577
- H1 central/greedy-AR cosine: 0.852671
- H1 P/Znext top-1 agreement: 0.400879

## Data and execution

- WikiText-103 raw UTF-8 train and validation bytes only
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed `1337 + 999`
- context 256; stride-one training anchors
- monitoring horizons 1--4; validation boundary stride 4
- effective batch 64; physical microbatch 16
- strict float32; TF32 disabled
- AdamW, peak LR `3e-4`, global clip norm 1.0
- 1000 updates on the fixed 6000-update schedule
- reports at 0, 1, 50, 100, then every 100 through 1000
- EMA is an exact online copy through update 100 and uses decay 0.99 after
  update 100
- primary validation and inference use the full EMA model

## Structural checks

- P-detached and ordinary forward values are bit-identical;
- successor closure has no gradient edge through P into hidden phase;
- CE retains its ordinary attached P graph;
- joint closure has finite nonzero gradients to Znext, successor Q/K/V,
  complex readout, memory phase, and online encoder through attached memory;
- selected y is exactly online P argmax;
- EMA canonical re-decode returns y as top-1;
- EMA targets and EMA parameters remain gradient-free;
- changing the held-out corpus byte changes only the CE label;
- literal EMA Encode(prefix,y) matches batched canonical action encoding;
- H1 central and greedy-AR CE logits remain structurally identical.

## Preregistered success criteria

Primary success at step 1000 requires all of:

- H1 Znext/canonical relative MSE below 0.996537;
- H1 Znext/canonical cosine above 0.061611;
- H1 P/Znext top-1 agreement at least 0.817871;
- H1 canonical-to-Znext KL at most 0.715272;
- H1 validation NLL below 2.347617;
- H1 central/greedy-AR max logit error below `5e-4`.

H2--H4 central/greedy-AR agreement, H2--H4 P/Znext agreement, sharpness,
gradient norm, exact block agreement, and monotonicity are supporting metrics.
The result must report hidden agreement and token-basin agreement separately.

Metrics are TSV and interpretation is Markdown. Checkpoints and smoke
artifacts remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_post_update_full_read_h1_ema_self_redecode_kl_mse_pdetach_ce_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2 \
  --record-dir \
    experiments/records/EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-self-redecode-kl-mse-pdetach-ce-13m \
  --output-dir \
    outputs/experiments/EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-self-redecode-kl-mse-pdetach-ce-13m
```

## Evidence boundary

This fixed-seed experiment tests greedy self-selected H1 closure. It does not
establish sampled-distribution equivalence and does not access the test split.
