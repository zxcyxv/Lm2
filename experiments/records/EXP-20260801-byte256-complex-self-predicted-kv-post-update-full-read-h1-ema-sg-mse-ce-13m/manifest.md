# EXP-20260801 post-update full-read recurrence

## Status

- State: completed all 1000 optimizer updates
- Direct control:
  `../EXP-20260801-byte256-complex-self-predicted-kv-h1-ema-sg-mse-ce-13m/`
- Test split remains unmaterialized and unread.

### Execution result

- The primary endpoint did not pass: final EMA H2--H4 agreement was
  `0.242269`, below the direct control's `0.272135`.
- H1 validation NLL improved from `6.336482` to `1.394581`; H1 latent relative
  MSE improved from `1.000298` to `0.335577`.
- Final H2, H3, and H4 agreement were `0.466553`, `0.152832`, and `0.107422`.
- Exact four-token block agreement was `0.009277`, below the control's
  `0.024658`.
- Write-enabled H2--H4 agreement remained above no-write agreement:
  `0.242269` versus `0.177897`.
- The registered residual/Shapley audit is in
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-residual-shapley-audit/`.

## Question

Does making the predicted next hidden a coherent reread of the complete
post-innovation memory improve central/greedy-AR agreement over the previous
split-query residual construction?

The sole mechanism change is the construction of the recurrent hidden.  The
current-token CE readout remains innovation-free and unchanged at identical
weights.

## Fixed recurrence and decoder contract

For recurrent hidden `Z[j]` and memory `M[j]`:

```text
Zrot       = Rh(Z[j])
Mrot       = Rm(M[j])
P[j]       = O(Read(Q(Zrot), Mrot))
W[j]       = K(P[j]) outer conj(V(P[j]))
M[j+1]     = Mrot + W[j]
Z[j+1]     = O(Read(Q(P[j]), M[j+1]))
```

- `P[j]` is the only central state sent to the exact-inverse decoder for the
  current future position and receives CE.
- `Z[j+1]` is not decoded at that position.  It is the latent-regression target
  and occupies the same recurrent-hidden slot on the next step that the
  initial encoded anchor occupied on the first step.
- The next step recomputes all Q/K/V projections from its own continuous
  state; no observed, held-out, sampled, or generated token enters the central
  recurrence.
- There is no beta, temperature, gate, residual prior, independent vocabulary
  head, decay coefficient, or L2 output normalization.
- Encoder and analytic decoder retain their reversible inverse construction;
  the central recurrence itself is not required to be invertible.

The direct control instead used:

```text
Z[j+1] = P[j] + O(Read(Q(P[j]), W[j]))
```

That mixes an old-query read of `Mrot` with a new-query read of `W[j]`.  The
new recurrence uses the new query on the complete updated memory.

## Loss and EMA target

```text
L = CE(Decode(prefix, P[0]), B)
    + relative_MSE(Z[1], sg(E_EMA(prefix, B)))
```

- Training horizon is one; longer horizons are target-free compositions used
  for evaluation.
- The future byte appears only as the CE label and in the gradient-free EMA
  coordinate target.  It never updates central memory.
- The EMA is an exact online copy through update 100.  From update 101 onward,
  full-model EMA decay is 0.99.
- Primary inference uses the complete EMA encoder, transition, inverse decoder,
  and tied simplex logits.  Online metrics remain secondary diagnostics.

## Data and fixed execution

- WikiText-103 raw UTF-8 train and validation bytes only
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed `1337 + 999`
- context 256; training horizon 1; stride-one training anchors
- monitoring horizons 1--4; validation boundary stride 4
- fixed 64 validation examples and 4096 evaluated boundaries
- effective batch 64; physical microbatch 16
- strict float32; TF32 disabled
- AdamW, peak LR `3e-4`, clip norm 1.0
- 1000 updates on the fixed 6000-update schedule
- reports at 0, 1, 50, 100, then every 100 updates through 1000

## Structural preflight

- At identical parameters, old and new modes must produce exactly the same
  `P[0]`, CE logits, innovation, and updated memory.
- The new recurrent hidden must equal a direct `Q(P[0])` read of the complete
  updated memory.
- Held-out future-byte mutations must not affect any central state or logits.
- H1 central and greedy-AR logits must agree within `5e-4` before feedback.
- EMA targets and EMA parameters must remain gradient-free.
- CE and H1 regression must retain finite nonzero gradients to the online
  encoder, Q/K/V projections, phases, and complex readout.

## Comparison and success criteria

The fixed-seed EMA direct control ended with:

- H2--H4 token agreement: `0.272135416667`
- exact H1--H4 block agreement: `0.024658203125`
- H2, H3, H4 agreement: `0.5078125`, `0.18798828125`, `0.12060546875`
- no-write H2--H4 agreement: `0.171630859375`
- H1 validation NLL: `1.393542081118`

Primary success requires final EMA H2--H4 agreement to be strictly above
`0.272135416667`.  The following are also required or reported separately:

- H1 validation NLL improves from initialization.
- write-enabled H2--H4 agreement exceeds no-write agreement.
- H1 central/AR maximum logit error remains below `5e-4`.
- exact block and each of H2, H3, H4 are compared with the control without
  suppressing regressions.
- monotonicity is reported over steps 200, 300, ..., 1000: both the complete
  H2--H4 sequence and whether every adjacent change is nonnegative are
  recorded.  Monotonicity is diagnostic rather than silently folded into the
  final endpoint criterion.

Metrics are TSV and interpretation is Markdown. Checkpoints and smoke
artifacts remain ignored.

## Agreement trajectory

The post-warm-start H2--H4 sequence was:

```text
step 200  0.193522
step 300  0.220703
step 400  0.246501
step 500  0.240560
step 600  0.247070
step 700  0.247152
step 800  0.242350
step 900  0.242676
step 1000 0.242269
```

It was not monotonic. H2 alone increased at every registered step from 200
through 1000, while H3/H4 and exact-block agreement did not; deeper rollout
degradation offset the H2 improvement.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_post_update_full_read_h1_ema_sg_mse_ce_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2 \
  --record-dir \
    experiments/records/EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-sg-mse-ce-13m \
  --output-dir \
    outputs/experiments/EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-sg-mse-ce-13m
```

## Evidence boundary

This is one fixed-seed validation experiment. It measures greedy
central/greedy-AR equivalence, not sampled joint-distribution equivalence, and
does not access the test split.
