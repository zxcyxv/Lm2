# EXP-20260802 final-carrier boundary post-norm H16 CE

## Status

- State: stopped after the step-300 report due to numerical/optimization divergence
- Intermediate-variable post-norm control:
  `../EXP-20260802-byte256-complex-self-predicted-kv-urm-postnorm-h16-ce-only-attached-stride16-13m/`
- Unnormalized H16 control:
  `../EXP-20260802-byte256-complex-self-predicted-kv-post-update-full-read-h16-ce-only-attached-stride16-13m/`
- Test split remains unmaterialized and unread.

The one-update smoke artifact is outside the repository. It measured raw P
RMS `0.0024..0.0080`, successor Z RMS `0.976..0.996`, and successor memory
RMS near `1.000`. Initial block NLL was `6.494108`, versus `45.232199` for the
intermediate-norm control. Step-1 raw global gradient norm was `25211.460938`,
versus `290453.3125` for that control. Future-input independence, causal
decoding, and every registered gradient path passed preflight.

The registered full run was interrupted during the update after its step-300
report. At step 300, raw global gradient norm reached `604649216`, block NLL
reached `223.148930`, and H1 logit RMS reached `373.724863`, despite both
recurrent carrier fields remaining at RMS approximately one. The partial TSV
and step-300 checkpoint are retained. This is a failed stability criterion,
not a completed 1000-step comparison.

## Question

Does applying fixed post-RMS normalization only to the complete successor
carrier, after one central transition has finished, improve H16 optimization
over normalizing P and memory inside the transition?

The completed intermediate-norm control reached block NLL `3.136612`. Its raw
global gradient norms were `765.149048` at step 50, `236.737488` at step 100,
and `10.558392` at step 1000. The unnormalized control became unstable and was
stopped after step 300.

## Registered computation

```text
S0 = write(encoded_prefix)
Z0 = encoded_prefix

for j in 1..16:
    Zbar_j = R Z_(j-1)
    Sbar_j = U S_(j-1)
    P_j = O(Read(Q(Zbar_j), Sbar_j))
    W_j = K(P_j) V(P_j)^dagger
    Sraw_j = Sbar_j + W_j
    Zraw_j = O(Read(Q(P_j), Sraw_j))

    S_j = ComplexRMSNorm(Sraw_j)
    Z_j = RMSNorm(Zraw_j)

[P_1, ..., P_16]
    -> one exact-inverse causal decode
    -> sixteen parallel token-logit rows

loss = mean(CE_1, ..., CE_16)
```

`P_j`, `W_j`, `Sraw_j`, and `Zraw_j` are not post-normalized before the raw
successor has been computed. Only the two heterogeneous fields of the final
recurrent carrier are normalized at the recurrence boundary. The two fields
use separate fixed non-affine RMS metrics rather than inventing an
unregistered relative weighting between a complex matrix and a real hidden
vector. This is a product-state boundary norm, not a joint concatenated norm.

The pre-existing Q/K/V input RMSNorms remain unchanged. No
`1/sqrt(write_count)` or other count-dependent read scaling is present. The
central and decoder paths are fully attached. There is no MSE, KL, EMA,
detach, token feedback, beta, independent head, or future-token input.

## Fixed protocol

- WikiText-103 raw UTF-8 train and validation bytes only
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- seed 1337; validation seed `1337 + 999`
- context 256; required window 272
- training/evaluation anchor stride 16
- 16 anchors by 16 horizons = 256 CE labels per sequence
- effective batch 64; physical microbatch 16
- strict float32; TF32 disabled
- AdamW, peak LR `3e-4`, clip norm 1.0
- 1000 updates on the fixed 6000-update schedule
- reports at 0, 1, 50, 100, then every 100 through 1000
- online model only

## Comparisons and success criteria

Preflight must establish exact future-input independence, causal decoder
triangularity, and finite nonzero H16 gradients to Q/K/V, complex readout,
phases, and the online encoder. It must show that P remains an unnormalized
decoder-facing readout while both successor carrier fields have bounded RMS
before they are reused.

Every horizon NLL must improve from initialization and final block NLL must be
below `4.0`. The positional hypothesis is supported more strongly if the
step-50 and step-100 raw global gradient norms are below `765.149048` and
`236.737488`, respectively, without final block NLL exceeding `3.136612`.
Gradient norm, carrier RMS, innovation energy, and logit RMS must remain finite
through H16. Conflicting metric directions will be retained rather than
collapsed into a single verdict.

Metrics are TSV and interpretation is Markdown. Checkpoints and smoke
artifacts remain ignored.

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_byte256_complex_self_predicted_kv_urm_boundary_postnorm_h16_attached_stride16_ce_only_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000 \
  --eval-examples 64 --eval-microbatch 2
```

## Evidence boundary

This is an H16 boundary-normalization ablation. It does not establish H32 or
H128 stability, sampled block quality, AR equivalence, a globally contractive
transition Jacobian, or test performance.
