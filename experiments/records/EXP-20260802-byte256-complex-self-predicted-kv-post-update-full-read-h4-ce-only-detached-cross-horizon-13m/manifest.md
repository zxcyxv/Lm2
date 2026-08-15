# EXP-20260802 detached-cross-horizon four-step central CE

## Status

- State: stopped by user after the step-100 report on 2026-08-02 UTC
- Matched attached-gradient control:
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h4-ce-only-13m/`
- Test split remains unmaterialized and unread.

## Question

With the normalized H4 CE-only architecture otherwise unchanged, what happens
to optimization, rollout NLL, peak VRAM, and step time when every synthetic
future slot consumes the preceding state values but cannot send its loss
gradient into preceding synthetic slots?

## Exact gradient contract

Forward values must be identical to the attached control. For horizon `j`:

- the numerical recurrent hidden and complex memory produced by `j-1` are the
  inputs to central transition `j`;
- those recurrent inputs are detached before transition `j`;
- decoder slot `j` sees the numerical earlier `P` slots causally;
- gradients from decoder slot `j` do not enter earlier `P` slots through their
  decoder key/value use;
- `CE_j` still trains the current read-side transition parameters, the decoder
  parameters used at `j`, and the shared real-prefix path;
- shared parameter updates remain shared across horizons. Only synthetic
  cross-horizon activation paths are cut.

Because `P_j` is read before innovation `W_j` is written, `W_j` affects only
later horizons. Exact boundary detachment therefore makes an H4-only loss have
zero gradient to K/V writes from H1--H3. Under the full four-horizon mean, K/V
still receive a gradient through the H1 initialization write. This is an
intentional measured consequence of the requested detach, not silently
replaced by a surrogate gradient.

This is not latent-target detachment, token feedback, teacher forcing, or
truncated data input. No corpus-future byte enters the central recurrence.

## Registered computation and objective

```text
state_0 = initialize(encoded prefix)

for j in 1..4:
    input_j = state_(j-1)                    if j == 1
              detach(state_(j-1))            otherwise
    P_j, state_j = central_transition(input_j)

[P_1, P_2, P_3, P_4]
    -> one exact-inverse causal decode whose past-slot state gradients detach
    -> four parallel logit rows

loss = mean(CE_1, CE_2, CE_3, CE_4)
```

Everything else matches the attached control: post-update full read,
`1/sqrt(write_count)` read scaling, stride-one 256 training boundaries,
four horizons, one rank-one innovation per transition, no MSE/KL/EMA/beta,
and no independent vocabulary head.

## Fixed protocol

- WikiText-103 raw UTF-8 train and validation bytes only
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed `1337 + 999`
- context 256; training anchor stride 1; validation anchor stride 4
- effective batch 64; physical microbatch 16
- 256 boundaries and 1024 CE labels per sequence
- strict float32; TF32 disabled
- AdamW, peak LR `3e-4`, clip norm 1.0
- 1000 updates on the fixed 6000-update schedule
- reports at 0, 1, 50, 100, then every 100 through 1000
- online model only

## Comparisons and success criteria

Preflight must establish:

- attached and detached forward read states, recurrent states, memories, and
  logits agree within `5e-5`;
- an H4-only token loss has zero gradient to decoder input slots H1--H3 and a
  finite nonzero gradient to H4;
- an H2 central read loss has no gradient to the initial root through H1 but
  has finite nonzero gradients to the shared central parameters;
- H4-only K/V gradients are zero, while its read-side gradients are finite and
  nonzero;
- the full mean CE retains finite nonzero gradients to Q/K/V, complex readout,
  phases, decoder, and encoder parameters;
- future-token leakage remains exactly zero.

At step 1000, report all per-horizon and block NLLs without suppressing a
negative result. The attached control reached block NLL `2.245606` and peak
VRAM `7507823104` bytes in `1376.815886` seconds. A peak below that byte count
is the directional memory criterion. Block NLL below 3.0 and improvement at
every horizon are minimum optimization criteria, not equivalence claims.

Metrics are TSV and interpretation is Markdown. Checkpoints and smoke
artifacts remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_post_update_full_read_h4_detached_cross_horizon_ce_only_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Evidence boundary

This experiment isolates synthetic cross-horizon gradient routing at H4. It
does not test stride 128, H128, sampled generation, or the test split.

## Partial result

The run was intentionally stopped after step 100. At that checkpoint detached
training had block NLL `3.117875` versus `2.971892` for the attached control.
Its H1--H4 NLLs were all higher. Peak allocated VRAM was `7689229312` bytes
versus `7507823104`, so detach did not provide a memory reduction at H4 under
this implementation. These partial metrics remain evidence and are not a
step-1000 comparison.
