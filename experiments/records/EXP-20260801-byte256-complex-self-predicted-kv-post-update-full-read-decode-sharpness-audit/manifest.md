# EXP-20260801 P versus Znext decode-sharpness audit

## Status

- State: completed; all 16,384 registered rows audited
- Parent:
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-sg-mse-ce-13m/`
- Read-only audit of the parent's step-1000 full-model EMA.
- No optimizer update, checkpoint mutation, or test-split access.

### Result

- H1 P/Znext top-1 agreement was `0.400879`.
- The local pre-innovation A/P agreement was `0.151123`; current innovation
  entered P's basin on `0.305176` of rows and left it on `0.055420`, for a net
  same-basin gain of about 25.0 percentage points.
- H1 Znext mean max-softmax probability was only `0.460469`, below P's
  `0.594960`; only `0.130615` of Znext rows exceeded 99% confidence.
- Canonical exact-inverse hidden states had mean max probability `0.999973`
  with zero top-1 errors, so the lack of sharpness is not a decoder ceiling.
- The trained innovation therefore helps basin entry but does not establish
  a same-token, realization-sharp future hidden.

## Question

Does the trained model realize the intended separation below?

```text
Decode(P):      predictive language distribution
Decode(Znext):  same top-1 token, but realization-sharp distribution
```

The head has no explicit softmax module. Training uses raw tied-simplex logits
with cross-entropy, which applies log-softmax internally. This audit applies
float32 softmax to the same raw logits only for probability diagnostics.

## Measurements

For every fixed validation boundary and horizon, decode two complete future
tapes already produced by the matched evaluator:

- `P` tape: innovation-free CE states;
- `Znext` tape: recurrent post-update full-read states.

Record:

- top-1 agreement between `Decode(P)` and `Decode(Znext)`;
- a local pre-innovation control `A = Read(q(P),Mrot)`, decoded behind the
  same preceding `Znext` tape, so basin entry/exit caused by the current
  innovation can be counted;
- max-softmax probability and entropy for each decode;
- fraction of `Znext` rows with max probability at least `0.90`, `0.99`, and
  `0.999`, independent of top-1 agreement;
- probability assigned by `Znext` to the top-1 token selected by `P`;
- the same statistics split by P/Znext top-1 match versus mismatch.

H1 is additionally split by whether `Decode(P)` correctly selected the
observed byte `B`. This distinguishes two cases that the joint objective treats
differently:

- when P already selects B, MSE and CE favor the same token basin;
- when P does not select B, the MSE target `E(prefix,B)` and P's selected token
  point to different basins.

As a decoder calibration check, decode the canonical future encoder tape from
the held-out bytes. Its top-1 target retrieval must be exact. This canonical
tape is only a calibration reference and is not the P/Znext agreement target.

Primary interpretation uses H1 because it compares exactly one `P` state with
one `Znext` state behind the same prefix. H2--H4 are reported but each decode
contains its corresponding accumulated future tape.

There is no matched no-MSE retraining control in this audit. Consequently it
can establish where the MSE-trained `Znext` landed and whether the current
innovation entered or left P's basin, but cannot by itself identify MSE as the
causal reason relative to training the same architecture without MSE.

## Fixed evidence

- parent EMA checkpoint: step 1000
- parent's 64 stored validation starts, seed `1337 + 999`
- WikiText-103 byte validation split; context 256
- horizons 1--4; anchor stride 4; 4096 rows per horizon
- evaluation microbatch 2; strict float32; TF32 disabled

## Interpretation criteria

The intended uncertainty/realization separation is supported only if:

- H1 P/Znext top-1 agreement is high; and
- H1 `Znext` is materially sharper than `P`, approaching the canonical
  decoder's confidence, including within top-1-matching rows.

Low top-1 agreement rejects token identity. A `Znext` distribution no sharper
than `P` rejects the realization-sharp interpretation even when the top-1
happens to match. Results are TSV; interpretation is Markdown.

## Producer

```bash
python eval_byte256_complex_self_predicted_kv_decode_sharpness.py \
  --record-dir \
    experiments/records/EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-decode-sharpness-audit \
  --examples 64 --microbatch 2
```
