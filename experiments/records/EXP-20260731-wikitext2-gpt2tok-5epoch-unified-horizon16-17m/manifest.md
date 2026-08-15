# EXP-20260731 WikiText-2 unified five-epoch horizon audit

## Status

- State: preregistered before optimizer update 1
- Authorization: the user requested fresh five-epoch runs of the
  state-conditioned latent model, a size-matched Transformer, and upstream
  Grassmann Flow, followed by one common N-token perplexity evaluation
- Primary split: validation
- Test split: not used by this experiment
- Seed: 1337 for initialization and train-chunk order
- Evaluated checkpoint: epoch 5 `last.pt`, never validation-selected `best.pt`

### Exact-loss implementation update after epoch 1

Epoch 1 completed under the original materialized implementation in
`927.8569` seconds. Before epoch 2 completed, the user authorized
implementation-only optimization with TF32/BF16 explicitly excluded.

The epoch-2--5 training path:

- computes canonical CE once for each of the 255 distinct decisions and then
  gathers scalar CE into the identical overlapping anchor/horizon weighting
- omits the Open-H1 vocabulary projection because it is exactly canonical H1
  and enters neither Open NLL nor closure
- shares one Open H2--H4 log-softmax between CE and KL
- omits the current-online closure diagnostic during training only; full
  validation diagnostics remain unchanged
- uses physical microbatch 8 with four accumulations instead of physical
  microbatch 4 with eight accumulations; the effective batch, sample order,
  per-example operations, loss mean, optimizer updates, and LR schedule are
  unchanged, and a deterministic gradient-equivalence test covers the two
  accumulation groupings

The scalar loss components and all parameter gradients were compared against
the materialized path on a deterministic small model with a distinct EMA
teacher. All thirteen state-conditioned tests pass. The epoch-1 model,
full-model EMA, optimizer, and five-epoch scheduler states are resumed from
the epoch boundary; no 20-epoch-schedule checkpoint is reused. Epochs 2--5
record `NaN` only for the omitted train-time online-closure diagnostic.

The optimized isolated full-size microbatch-4 backward peaks at `3.5486 GiB`.
An isolated microbatch-16 backward peaks at `13.7742 GiB`, but the complete
resume process also holds AdamW state and accumulated gradients and failed
before update 293 while requesting another `774 MiB` with `605.81 MiB` free.
The failed attempt changed no model, optimizer, scheduler, metric, or
checkpoint state and is retained in `failed_microbatch16_resume.tsv`.
Production therefore uses microbatch 8 with four accumulations and a larger
capacity margin. All paths remain strict FP32 with TF32 disabled.

Epoch 2 completed in `755.8641` seconds, an `18.54%` wall-time reduction from
epoch 1 rather than the hoped-for twofold speedup. The vocabulary GEMMs and
the separately routed supervised/closure backward traversals remain the
dominant arithmetic after materialization was reduced. Before epoch 3 made an
update, the run was paused once more to add visible `tqdm` train and validation
progress, update count, running loss, LR, throughput, and ETA. This logging
does not enter model computation or optimizer state.

## Question

After the same five passes over WikiText-2, does the state-conditioned
continuous recurrence retain competitive ordinary next-token likelihood, and
does its inference-equivalent open rollout lose no more likelihood over
multiple generated positions than size-matched token-feedback baselines?

## Models

| name | fixed architecture | expected parameters |
|---|---|---:|
| `latent` | width 336, two reversible causal encoder blocks, exact inverse decoder, state-conditioned two-stage Givens transition and innovation, RMS-tied head | 17,696,112 |
| `transformer` | upstream `SmallTransformer`, width 256, six blocks, eight attention heads, FFN 1,024, tied head | 17,670,400 |
| `grassmann` | upstream `GrassmannGPTv4`, width 256, six blocks, reduced dimension 32, windows `[1,2,4,8,12,16]`, tied head | 17,695,168 |

The baseline architectures are loaded from the preserved upstream snapshot at
`outputs/external/grassmann-flows`; their model definitions are not copied
into an experiment wrapper. The latent loss remains its registered canonical
H1--H4 CE plus H2--H4 open CE and EMA behavioral closure. Both baselines use
ordinary next-token CE. These objective differences are part of the compared
training methods and are reported rather than hidden.

## Data and shared optimization controls

- Hugging Face `wikitext/wikitext-2-raw-v1`
- `GPT2Tokenizer.from_pretrained("gpt2")`, vocabulary 50,257
- discard empty rows, newline-join, tokenize each complete split, truncate to
  non-overlapping 256-token chunks
- expected chunks: train 9,343 and validation 965
- fresh initialization; no checkpoint continuation
- exactly 5 epochs and 292 optimizer updates per epoch
- effective batch 32; latent physical microbatch 4, baselines physical batch 32
- AdamW, betas `(0.9, 0.95)`, weight decay `0.01`, peak LR `3e-4`
- per-update `CosineAnnealingLR`, `T_max = 5 * 292 = 1,460`
- gradient clipping 1.0
- strict float32 CUDA execution with TF32 disabled
- full ordinary validation after every epoch

Registered tokenized-data provenance:

| split | raw rows | nonempty rows | untruncated tokens | stored tokens | chunks | token int32 SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| train | 36,718 | 23,767 | 2,391,884 | 2,391,808 | 9,343 | `ba036632a98cebca20c2cb678d3547b7f607f4cb8e4e540814bcba18bd44af89` |
| validation | 3,760 | 2,461 | 247,289 | 247,040 | 965 | `e2663296a2cd2e1d6b72deb89afb76102d0dffbf2d6357af9042cb700594d02c` |

## One common N-token evaluation

Every validation chunk is split into the same 240-token literal prefix and
16-token held-out suffix. The evaluator computes token CE before argmax and
reports per-horizon and cumulative PPL for `N = 1, 2, 4, 8, 16`.

- `teacher`: each suffix decision receives its literal gold-token context.
- `open`: only the first 240 tokens are gold. Transformer and Grassmann feed
  back their greedy token; the latent model feeds back only its continuous
  transition state and performs one joint exact-inverse readout after the
  latent tape is complete.
- `open_excess_nll = open_nll - teacher_nll` is the primary degradation
  quantity. Open PPL is a fixed-reference rollout score, not the ordinary
  chain-rule perplexity of a token-conditioned AR distribution.

All three models are scored against identical suffix token IDs by the same CE
accumulator. No model-specific evaluator, sampling temperature, best-checkpoint
selection, or generated-text quality proxy enters the primary table.

## Success criteria

- all three fresh runs complete five epochs without non-finite loss
- the latent H1 open PPL is no worse than both size-matched baselines
- at cumulative N=16, latent open excess NLL is no greater than either
  baseline's open excess NLL
- report every registered horizon even if evidence conflicts with the two
  criteria

## Producers

```bash
PYTHONPATH=src python train_wikitext2_unified_5epoch.py --model latent
PYTHONPATH=src python train_wikitext2_unified_5epoch.py --model transformer
PYTHONPATH=src python train_wikitext2_unified_5epoch.py --model grassmann
PYTHONPATH=src python eval_wikitext2_unified_horizon_ppl.py
```
