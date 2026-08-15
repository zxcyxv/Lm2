# EXP-20260801 complex self-predicted KV central transition

## Status

- State: aborted after the step-250 report by user correction
- Reason: the learned innovation coefficient was a premature temperature
  experiment and was not part of the requested base mechanism
- Authorization: user requested implementation and training after fixing the
  central recurrence to consume only its own predicted hidden state
- Test split remains unmaterialized and unread

## Question

Can one jointly learned encoder, lightweight complex central recurrence, and
analytic inverse decoder learn the one-step conjugacy

`T(E(x_<=t)) ~= E(x_<=t+1)`

well enough that self-composition of `T` follows the same model's ordinary
greedy decode--append--re-encode trajectory through four steps?

The central recurrence receives neither a held-out future byte nor a sampled
or greedy byte.  Every rank-one KV write is generated only from the central
layer's own continuous prediction.  The ordinary AR path is an evaluation
reference, not a teacher model or a distillation target.

## Fixed model

- byte vocabulary: 256
- width: 1344; two reversible causal encoder blocks
- decoder: the encoder's analytic exact inverse
- no independent vocabulary head
- one fixed regular-simplex byte codebook shared by input embeddings and raw
  tied output scoring
- raw logits: `16 * decoded_hidden @ codebook.T`
- no output-vector L2 normalization, radius loss, latent normalization, EMA,
  teacher model, distillation, sampled branch, token-conditioned update, or
  future-token input

The reversible contract applies only to encoder and decoder.  The central
transition is not required to be injective or reversible.

## Complex central recurrence

Real hidden coordinates are paired as complex modes; implementation may keep
real and imaginary parts explicitly or use complex64 internally without an
angle/`atan2` coordinate conversion.

Fixed dimensions:

- heads: 8
- complex key dimension per head: 16
- complex value dimension per head: 31
- one learned global hidden-channel phase tape
- one learned global memory-row phase tape
- no decay factor

For root hidden `h` initialize matrix memory from that same known root:

`S_0 = k(h) v(h)^dagger`.

One central step is:

1. rotate `h` and `S` with learned unit-modulus phases;
2. read rotated memory to obtain a preliminary self-prediction `h_pre`;
3. form `I = k(h_pre) v(h_pre)^dagger`;
4. update full memory `S_next = S_rot + I`;
5. read the same self-generated innovation to obtain `delta`;
6. use `h_full = h_pre + delta` as the next recurrent hidden state;
7. use `h_read = h_pre + beta * delta` for token decoding.

`beta = sigmoid(beta_logit)` is one learned real scalar initialized to `0.5`.
It scales only the CE readout.  It does not scale `S_next` or `h_full`, so the
free scale of the KV projections cannot trivially cancel it in the online
latent-closure path.

## Objective

Training supervises every stride-one prefix anchor at horizon one only.

`L_CE = CE(raw_tied_logits(D(h_read)), byte_(t+1))`

`L_H1 = ||h_full - h_(t+1,online)||^2 / ||h_(t+1,online)||^2`

`L = L_CE + 1.0 * L_H1`.

The future state is produced by the same online encoder pass and remains
attached.  Gradients may reshape both sides of the latent coordinate system.
The future state is a loss target only and is never supplied to the central
recurrence.

At monitoring time the central layer self-composes for four steps, produces
all four readout states, and the inverse decoder evaluates the complete causal
tape in one batched call.

## Controls and evidence scope

- historical producer:
  `EXP-20260801-byte256-simplex-tied-self-composition-h1-ce-13m`; its
  normalized readout and radius loss differ and are retained as scoped
  evidence rather than treated as a perfectly matched control
- checkpoint-time functional ablation with the self-generated rank-one write
  removed while keeping the learned unitary rotations and projections
- checkpoint-time full-gain readout (`beta=1`) versus learned tempered readout
- deterministic step-0 initialization

A separately trained parameter-matched no-KV control is required before an
architecture-level performance claim.  The first run is a mechanism test.

## Data and execution

- source bytes and provenance:
  `data/wikitext103_bytes/provenance.tsv`
- train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- train and validation only; test is not materialized
- seed: 1337; fixed validation seed: `1337 + 999`
- context: 256 bytes
- training horizon: 1; stride-one anchors
- monitoring horizon: 4; boundary stride 4
- effective batch: 64; initial physical microbatch: 16
- strict float32 with TF32 disabled
- AdamW, peak LR `3e-4`, gradient clip 1.0
- initial run: 1000 optimizer updates on the fixed 6000-update schedule
- reports: 0, 1, 50, 100, 250, 500, 750, 1000

If microbatch 16 does not fit, reducing only physical microbatch while
preserving effective batch, sampled windows, seed order, and optimizer update
semantics is an allowed operational correction and must be recorded.

## Structural preflight

- the raw tied codebook retrieves every one of its own byte vectors
- no independently parameterized vocabulary-scoring projection exists;
  internal complex query/key projections may coincidentally have output width
  256 and are distinguished by their role rather than their tensor shape
- changing held-out future bytes leaves the complete self-fed four-step tape
  unchanged
- horizon-one parallel and AR proposals agree before hard-token feedback
- CE has finite nonzero gradients to encoder, complex projections, phases,
  innovation readout, and `beta`
- attached H1 loss has finite nonzero gradients to both the predicted state
  path and the online future encoder state
- every reported state, memory energy, phase, and gradient is finite

## Step-1000 success criteria

All of the following are required for the initial mechanism criterion:

- validation H1 NLL is lower than deterministic step 0
- validation attached H1 relative MSE is lower than deterministic step 0
- H2--H4 self-composition/greedy-AR byte agreement is at least `0.50`
- four-horizon mean self-composition/greedy-AR state cosine is at least `0.75`
- full-gain innovation and learned-beta readouts are observably distinct
- learned `beta` remains strictly between `0.01` and `0.99`
- no-future-leak and one-step-path preflight tolerances continue to pass

Numeric results go to TSV.  Interpretation and any failure decomposition go
to Markdown.  Checkpoints and smoke artifacts remain under ignored
`outputs/` paths.
