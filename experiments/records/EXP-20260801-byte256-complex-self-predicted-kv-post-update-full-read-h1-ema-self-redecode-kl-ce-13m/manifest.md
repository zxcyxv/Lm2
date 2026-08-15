# EXP-20260801 EMA self-redecode KL closure

## Status

- State: preregistered; not yet executed
- Direct MSE control:
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-sg-mse-ce-13m/`
- Decode-sharpness audit:
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-decode-sharpness-audit/`
- Test split remains unmaterialized and unread.

## Question

Can decoder-visible forward KL replace hidden-space MSE and make the recurrent
`Znext` decode into the same, realization-sharp token basin selected by the
innovation-free CE state `P`?

## Fixed recurrence

The post-update full-read recurrence is unchanged:

```text
P       = O(Read(Q(Zrot), Mrot))
W       = K(P) outer conj(V(P))
Mnext   = Mrot + W
Znext   = O(Read(Q(P), Mnext))
```

- Only `P` enters the online exact-inverse decoder for ordinary corpus CE.
- `Znext` remains the next recurrent hidden.
- No observed, selected, generated, or held-out token enters the central
  recurrence or writes central KV memory.
- There is no latent MSE, beta, temperature, gate, decay, residual prior,
  L2 normalization, or independent vocabulary head.

## Self-selected EMA redecode path

For each anchor:

```text
online P logits -> greedy token y
y appended to the literal prefix -> EMA encoder -> canonical state H(y)
H(y) -> EMA exact-inverse decoder -> canonical log-softmax R
online Znext -> the same frozen EMA exact-inverse decoder -> log-softmax S
closure = KL(R || S)
```

The selected token `y` is the model's own online P argmax, not the corpus
future byte. It is used only to construct the closure target and is never fed
to the central recurrence.

The exact-inverse decoder has no independent parameters: it runs the EMA
encoder blocks in analytic reverse. Therefore a full-model EMA supplies both
the EMA encoder and EMA inverse decoder. EMA parameters are frozen, while
autograd through the frozen EMA decoder remains enabled with respect to its
online `Znext` input.

The canonical branch is evaluated under `no_grad`. The student branch is not:
KL gradients pass through the frozen EMA inverse decoder into online `Znext`,
the complex transition, and the online encoder root, but never create EMA
parameter gradients.

## Objective

```text
L = CE(online Decode(P), observed B) + KL(R || S)
```

- CE weight: 1.0
- forward-KL weight: 1.0
- native softmax temperature: 1.0
- training horizon: H1 only
- the observed B participates only as the ordinary CE label
- no hidden-state regression term remains

## EMA schedule and inference

- The EMA is an exact online copy through update 100.
- Beginning at update 101, full-model EMA decay is 0.99.
- Primary validation and inference use the complete EMA snapshot.
- Online-model metrics against the frozen EMA redecode path are retained as
  secondary diagnostics.

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
- AdamW, peak LR `3e-4`, clip norm 1.0
- 1000 updates on the fixed 6000-update schedule
- reports at 0, 1, 50, 100, then every 100 through 1000

## Structural checks

- selected `y` exactly equals the online P argmax;
- EMA canonical re-decode returns `y` as top-1 with near-simplex confidence;
- EMA canonical states/logits and all EMA parameters are gradient-free;
- KL has finite nonzero gradients to online `Znext`, Q/K/V, complex readout,
  phases, and encoder root;
- no EMA parameter accumulates a gradient;
- changing the corpus future byte changes CE labels only and leaves P,
  selected y, canonical EMA distribution, Znext, and student distribution
  unchanged;
- literal `Encode_EMA(prefix,y)` matches the batched canonical action state;
- H1 central/greedy-AR CE logits remain structurally identical.

## Fixed comparison and success criteria

The MSE control ended at:

- H1 P/Znext top-1 agreement: `0.400879`
- H1 Znext mean max-softmax probability: `0.460469`
- H1 Znext 99%-confidence fraction: `0.130615`
- H2--H4 central/greedy-AR agreement: `0.242269`
- H1 validation NLL: `1.394581`

Primary success at step 1000 requires:

- H1 P/Znext top-1 agreement above `0.400879`;
- H1 Znext mean max probability above `0.460469`;
- H1 Znext 99%-confidence fraction above `0.130615`;
- EMA canonical-to-Znext forward KL below canonical-to-P forward KL;
- H1 validation NLL improves from initialization;
- H1 central/greedy-AR max logit error remains below `5e-4`.

At every monitoring horizon, P/Znext top-1 agreement, Znext mean maximum
softmax probability, and the Znext 99%-confidence fraction are recorded.
H2--H4 agreement, exact block agreement, latent cosine/MSE, and monotonicity
are supporting metrics. A sharp self-selected state is not treated as a
calibrated corpus conditional distribution.

Metrics are TSV and interpretation is Markdown. Checkpoints and smoke
artifacts remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_post_update_full_read_h1_ema_self_redecode_kl_ce_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2 \
  --record-dir \
    experiments/records/EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-self-redecode-kl-ce-13m \
  --output-dir \
    outputs/experiments/EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-self-redecode-kl-ce-13m
```

## Evidence boundary

This fixed-seed experiment tests greedy self-selected behavioral closure. It
does not establish sampled-distribution equivalence and does not access the
test split.
