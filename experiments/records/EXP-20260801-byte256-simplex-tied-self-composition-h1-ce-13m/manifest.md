# EXP-20260801 byte-256 simplex-tied self-composition

## Status

- State: preregistered before optimizer update 1
- Authorization: user requested the equal-norm dot-product construction and
  approximate parameter matching to the preceding BPE checkpoint
- Test split is not materialized or evaluated.

Initial segment: completed through step 1000. The original strict absolute
criterion was missed; this is retained as evidence and is not interpreted as
an architecture-level failure. Step 1000 is only one sixth of the already
fixed 6000-step LR schedule, and the late trajectory was still improving.

Continuation through step 6000 was preregistered below before optimizer
update 1001.

## Question

Does replacing the unconstrained BPE-8192 tied readout with a fixed,
equal-norm byte-256 regular-simplex codebook make the decoder's continuous
output close to its greedy token embedding, and thereby make input-dependent
continuous self-composition track greedy AR generation through four steps?

## Data

- source: `Salesforce/wikitext`, `wikitext-103-raw-v1`
- encoding: literal UTF-8 byte values `0..255`; no tokenizer or special IDs
- train: 55,000,000 bytes, SHA-256
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- validation: 1,146,709 bytes, SHA-256
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- producer and provenance: `prepare_wikitext103_bytes.py` and
  `data/wikitext103_bytes/provenance.tsv`
- train and validation only; test is not materialized

## Model

- vocabulary/codebook: 256
- width 1344; two reversible encoder blocks; exact analytic inverse decoder
- total parameters `13,200,096`, versus preceding BPE total `13,199,424`
- trainable parameters `12,854,016`
- one fixed regular-simplex codebook shared by input embedding and output
  scoring; code vectors have norm one and pairwise dot product `-1/255`
- codebook and legacy learned RMSNorm are frozen
- logits: fixed-scale normalized dot product
  `16 * normalize(y) @ codebook.T`
- no independent LM-head or trainable vocabulary projection
- input-conditioned real-normal K generator, bottleneck 73, alpha zero,
  initial radius 0.99

Equal code norms imply

`argmax_v y dot e_v = argmin_v ||normalize(y) - e_v||^2`.

## Training objective

At every stride-one prefix anchor, train horizon one only:

`L = CE(logits(y_1), byte_(t+1)) + 10 * (||y_1|| - 1)^2`

where `y_1 = E^-1(K(E(x_t)) E(x_t))`. The radial term only places the
continuous decoder output on the codebook radius. There is no future-hidden
regression, latent MSE, h2--h4 CE, KL, innovation, branch, noise, or token
feedback in the trained path.

## Correct rollout and AR reference

The continuous path regenerates K from every previous continuous latent
output:

`u_0 = E(x_t)`

`u_(j+1) = K(u_j) u_j`.

Thus step two is `K_(t+1) K_t E(x_t)`, not a frozen `K_t^2`.

The AR path takes the model's greedy byte, appends it to its own history, and
canonically re-encodes it before generating the next K. Held-out future bytes
are not inserted into the AR state.

## Fixed protocol

- seed 1337; fixed validation seed `1337 + 999`
- context 256 bytes
- training: horizon one, anchor stride one
- monitoring: horizons one through four, boundary stride four
- effective batch 64, microbatch 16
- AdamW, peak LR `3e-4`, existing 6000-step schedule, clip norm 1.0
- 1000 updates; reports at 0, 1, 50, 100, 250, 500, 750, 1000
- 64 fixed validation examples
- strict float32; TF32 disabled

## Metrics and criteria

For each horizon record:

- `y_j` versus selected simplex embedding cosine, squared distance, and norm;
- continuous-composition versus canonical greedy-AR state cosine and relative
  MSE;
- continuous-composition versus greedy-AR token agreement;
- one-step closure and accumulated-path components separately.

The construction succeeds only if step 1000 has all of:

- lower validation h1 NLL than step 0;
- mean decoder/codebook cosine at least 0.90;
- mean decoder norm absolute error at most 0.10;
- h2--h4 continuous/AR token agreement at least 0.80; and
- mean continuous/AR state cosine at least 0.90.

Numeric metrics are TSV and interpretation is Markdown. Checkpoints remain in
ignored `outputs/` paths.

## Preregistered continuation before update 1001

The step-1000 result does not justify a failure conclusion merely because a
cosine is below one (or below the original absolute threshold). From step 500
to 1000, H1 decoder/codebook cosine increased from `0.396437` to `0.429469`
and H1 continuous/AR state cosine increased from `0.142194` to `0.250843`.
At step 1000 the LR was still `2.8482566e-4`, close to the `3e-4` peak of the
fixed 6000-step schedule.

The same run will therefore resume from `step1000.pt` through step 6000 with:

- identical model, codebook, data, seed, objective, optimizer state, batch,
  microbatch, and 6000-step LR schedule;
- the CPU sampling generator reconstructed by replaying the one `randint`
  draw made at each of updates 1--1000 (later checkpoints store its state);
- the same fixed validation starts and four-step self-composition/greedy-AR
  evaluator; and
- reports at 1250, 1500, and every 500 updates from 2000 through 6000.

The original absolute step-1000 criterion remains recorded separately. The
continuation trend criterion passes only if, from step 1000 to step 6000, all
of the following hold:

- H1 validation NLL does not increase;
- H1 and four-horizon-mean decoder/codebook cosine both increase;
- four-horizon-mean continuous/AR state cosine increases;
- H2--H4 continuous/AR greedy-token agreement increases; and
- mean decoder norm absolute error remains at most `0.10`.

This endpoint trend criterion tests whether additional optimization continues
to close the measured gap. It does not redefine cosine `1` as necessary, nor
does passing it by itself establish exact AR equivalence.
