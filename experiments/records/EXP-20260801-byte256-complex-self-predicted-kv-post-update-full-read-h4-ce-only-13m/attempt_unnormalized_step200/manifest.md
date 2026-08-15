# EXP-20260801 four-step central, parallel four-token CE

## Status

- State: preregistered; not yet executed
- Immediate joint-closure predecessor:
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-self-redecode-kl-mse-pdetach-ce-13m/`
- EMA H1 MSE control:
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-sg-mse-ce-13m/`
- Test split remains unmaterialized and unread.

## Question

Can direct four-token CE train four sequential target-free central transitions
to produce a useful four-position latent tape, without hidden regression,
decoder-distribution KL, EMA targets, or discrete token feedback?

## Registered computation

For every causal prefix boundary, the central layer runs sequentially four
times:

```text
state_0 = initialize(encoded prefix)

for j in 1..4:
    P_j       = O(Read(Q(Zrot_j), Mrot_j))
    W_j       = K(P_j) outer conj(V(P_j))
    M_j       = Mrot_j + W_j
    Z_j       = O(Read(Q(P_j), M_j))
    state_j   = (Z_j, M_j)
```

The four innovation-free predictions are then decoded together:

```text
[P_1, P_2, P_3, P_4]
    -> one exact-inverse causal tape decode
    -> four parallel logit rows
    -> mean CE against [B_1, B_2, B_3, B_4]
```

The central recurrence is sequential and lightweight. "Parallel" refers to
the single batched causal decoder evaluation of all four future positions;
there is no sequential token sampling or token re-encoding between positions.

The four corpus-future bytes are labels only. They never enter central hidden,
memory, K/V, decoder inputs, or an auxiliary target branch.

## Objective

```text
loss = mean_j CE(logits_j, B_j), j=1..4
```

- H1--H4 CE weights: equal through one mean
- MSE weight: 0
- KL weight: 0
- EMA: absent
- token re-encoding: absent
- beta/temperature/gate/decay/residual prior/L2 normalization: absent
- independent vocabulary head: absent
- recurrent hidden mode: post-update full read

## Fixed comparisons

The predecessor trained only H1 joint self-selected hidden MSE plus decoder KL
and ended at:

- H1 validation NLL: 2.215206
- H1 central/canonical cosine: 0.503774
- H1 P/Znext top-1: 0.854248
- H2/H3/H4 central/greedy-AR agreement:
  0.378418 / 0.298584 / 0.470703

The EMA MSE-only control trained H1, not a four-token CE block, and ended at
H1 validation NLL 1.394581. Neither predecessor recorded a directly matched
four-token mean validation NLL, so per-horizon and block NLL are primary for
this new comparison.

## Data and execution

- WikiText-103 raw UTF-8 train and validation bytes only
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed `1337 + 999`
- context 256; stride-one training boundaries
- training and monitoring horizons: 4
- validation boundary stride 4
- effective batch 64; physical microbatch 16
- 256 boundaries and 1024 CE labels per sequence
- strict float32; TF32 disabled
- AdamW, peak LR `3e-4`, global clip norm 1.0
- 1000 updates on the fixed 6000-update schedule
- reports at 0, 1, 50, 100, then every 100 through 1000
- primary inference model: online model; no EMA snapshot exists

## Structural checks

- training token CE has shape `[batch,256,4]`;
- the scalar objective is exactly the mean of all four CE planes;
- no hidden target receives a gradient;
- H4 CE has finite nonzero gradients to Q/K/V, complex readout, both phases,
  and the online encoder;
- changing all four held-out tail bytes changes labels but not the four central
  states or logits;
- perturbing P4 cannot change decoded H1--H3 logits but does change H4 logits;
- decoder readout at every horizon is the innovation-free P state;
- central recurrence consumes Znext and matrix memory only;
- no beta, residual prior, auxiliary loss, target model, or independent head
  exists.

## Metrics

Metrics record:

- H1, H2, H3, and H4 target NLL and accuracy;
- their four-token mean validation NLL and accuracy;
- central/greedy-AR token agreement by horizon;
- P/Znext token agreement and confidence by horizon;
- continuous/AR cosine and relative MSE by horizon;
- innovation and memory energy;
- throughput per unique input byte and per CE label.

## Preregistered success criteria

At step 1000:

- four-token mean validation NLL is below 3.0 and below initialization;
- every H1--H4 target NLL improves from initialization;
- every H1--H4 target accuracy exceeds random byte accuracy `1/256`;
- mean H2--H4 central/greedy-AR token agreement exceeds the joint predecessor
  mean of 0.382568;
- H1 central/greedy-AR max logit error remains below `5e-4`.

The per-horizon NLL curve, exact four-token block agreement, and agreement
monotonicity are reported even if the aggregate criteria pass.

Metrics are TSV and interpretation is Markdown. Checkpoints and smoke
artifacts remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_post_update_full_read_h4_ce_only_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2 \
  --record-dir \
    experiments/records/EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h4-ce-only-13m \
  --output-dir \
    outputs/experiments/EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h4-ce-only-13m
```

## Evidence boundary

This experiment tests deterministic four-position block prediction under
greedy evaluation. It does not establish sampled-distribution equivalence and
does not access the test split.
