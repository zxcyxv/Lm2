# EXP-20260801 complex self-predicted KV drift and margin audit

## Status

- State: completed; all registered checkpoints and 16,384 margin rows audited
- Parent:
  `EXP-20260801-byte256-complex-self-predicted-kv-full-innovation-h1-online-mse-ce-13m`
- This is a read-only post-hoc audit. It performs no optimizer update and does
  not alter any checkpoint.
- Test split remains unmaterialized and unread.

## Questions

1. Did the jointly trained online encoder move its fixed-prefix future-state
   coordinates enough between saved checkpoints to make target motion a
   plausible primary cause of the late H2--H4 agreement decline?
2. At step 1000, do central/greedy-AR token mismatches occur mainly at narrow
   AR decision margins, or do they require large latent/logit perturbations?
3. Does the innovation-updated H1 posterior decode the observed next byte as
   reliably as the exact canonical future state?

The audit decides whether an EMA-coordinate ablation is justified before
changing the parent objective. It does not train an EMA model.

## Fixed evidence and split

- checkpoints: steps 100, 250, 500, 750, and 1000 from the parent producer
- WikiText-103 raw-byte validation split only
- the parent's 64 stored validation starts, seed `1337 + 999`
- context 256, horizon 4, anchor stride 4, evaluation microbatch 2
- strict float32 and TF32 disabled
- byte targets are used only for accuracy diagnostics; no gradient is built

## Encoder-drift protocol

For every checkpoint, encode the identical 64 validation windows and retain
the canonical H1 future state at all 64 stride-four anchors. This gives 4096
fixed state rows per checkpoint.

For each adjacent checkpoint pair and each checkpoint versus step 1000,
record:

- raw rowwise cosine, relative MSE, and norm ratio;
- centered linear CKA;
- the optimal global orthogonal Procrustes alignment and its aligned cosine
  and relative MSE.

Raw motion that largely disappears under one global orthogonal alignment is
reported as coordinate gauge motion, not automatically as harmful target
motion. Drift that remains after alignment and lowers CKA is structural.

## Step-1000 margin protocol

Use the existing matched evaluator. The central path repeatedly consumes only
its own continuous prior/posterior state. The greedy-AR reference feeds back
only its own selected byte through canonical re-encoding; it never writes a
byte into central KV memory.

For every horizon and boundary, take greedy-AR logits as the reference and
central logits as the candidate. Record separately for matching and
mismatching rows:

- AR top-1 versus runner-up margin;
- candidate margin for the AR-selected byte;
- exact differential perturbation toward the candidate's strongest
  competitor and its excess over the AR decision margin;
- maximum absolute logit delta;
- central/AR state cosine and relative MSE.

Also record prior target accuracy, innovation-updated posterior target
accuracy, and canonical-gold target accuracy by horizon.

## Interpretation rules

- EMA-coordinate stabilization is a supported next ablation only if late
  checkpoint motion remains substantial after Procrustes alignment and CKA
  shows non-gauge representational change. Large raw drift alone is
  insufficient.
- A decoder-boundary account is supported when mismatches have materially
  narrower AR margins than matches while their state geometry remains close.
- A transition/self-composition account is supported when mismatches retain
  ordinary AR margins but have large state or differential-logit errors.
- Both mechanisms may be present. Temperature rescaling cannot change greedy
  argmax agreement and is outside this audit.

## Completion criteria

- every registered checkpoint loads the coefficient-free prior-readout model;
- all drift and margin metrics are finite;
- the algebraic flip test agrees with observed top-1 mismatch on every row
  without a logit tie;
- canonical-gold exact-inverse decoding retrieves its byte at every audited
  horizon;
- metrics are written to TSV and interpretation to Markdown.
