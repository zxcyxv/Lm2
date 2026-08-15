# EXP-20260801 matched sqrt(n) H1 CE-only control

## Status

- State: completed; 200-step registered run finished on 2026-08-02 UTC
- Direct four-horizon comparison:
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h4-ce-only-13m/`
- H1 MSE+CE comparison:
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-sg-mse-ce-13m/`
- Test split remains unmaterialized and unread.

## Question

Under the same normalized post-update full-read architecture, does H1 CE-only
reduce H1 NLL faster than equal-weight H1--H4 CE? This separates multi-horizon
gradient interaction from the auxiliary MSE used by the older H1 control.

## Registered computation and objective

```text
Mrot  = U(M0)
P1    = O(Read(Q(Zrot), Mrot)) / sqrt(1)
W1    = K(P1) outer conj(V(P1))
M1    = Mrot + W1
Z1    = O(Read(Q(P1), M1)) / sqrt(2)
loss  = CE(Decode(P1), B1)
```

The producer still computes `W1`, `M1`, and `Z1` through the shared transition,
but CE depends only on causal prior `P1`. There is no MSE, KL, EMA, beta,
residual prior, L2 normalization, token feedback, or independent vocabulary
head. No architecture or read-scaling mechanism differs from the active H4
run; training horizon and objective labels are the only changes.

## Fixed protocol

- WikiText-103 raw UTF-8 train and validation bytes only
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed `1337 + 999`
- context 256; stride-one training boundaries
- effective batch 64; physical microbatch 16
- 256 H1 CE labels per sequence
- strict float32; TF32 disabled
- AdamW, peak LR `3e-4`, clip norm 1.0
- 200 updates on the fixed 6000-update schedule
- reports at 0, 1, 50, 100, and 200
- online model only

## Comparisons and success criteria

At matched step 200:

- normalized H4 CE-only H1 NLL: `2.168539`
- unnormalized H1 EMA-MSE+CE H1 NLL: `2.063148`

Primary results are the full H1 NLL curve and step-200 differences against
both values. H1 target accuracy must exceed `1/256`, H1 NLL must improve from
initialization, and central/AR H1 logits must remain identical within `5e-4`.
The MSE comparison is explicitly not architecture-matched because it lacks
sqrt(n); it is retained only to determine whether a further matched MSE
control is warranted.

Metrics are TSV and interpretation is Markdown. Checkpoints and smoke
artifacts remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_post_update_full_read_sqrtn_h1_ce_only_13m.py \
  --steps 200 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Result

Matched validation H1 NLL favored H1 CE-only at every registered checkpoint:

| Step | H1 CE-only | H1--H4 CE-only | Delta |
|---:|---:|---:|---:|
| 50 | 2.531328 | 2.584717 | -0.053390 |
| 100 | 2.395369 | 2.489069 | -0.093700 |
| 200 | 1.879377 | 2.168469 | -0.289092 |

The rollout comparison went in the opposite direction at step 200. H1-only
reached block NLL 3.487789 with H2--H4 NLLs 3.828577, 4.050170, and 4.193032.
Four-horizon CE reached block NLL 2.769732 with H2--H4 NLLs 2.744624,
3.027310, and 3.138527. Thus the registered evidence shows an immediate-H1
versus recurrent-rollout trade-off; it does not show that downstream CE
improves H1 under equal horizon weighting.
