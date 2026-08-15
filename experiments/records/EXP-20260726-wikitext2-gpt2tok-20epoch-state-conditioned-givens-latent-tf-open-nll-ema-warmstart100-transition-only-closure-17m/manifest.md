# EXP-20260726 WikiText-2 state-conditioned latent TF EMA, 20 epochs

## Status

- State: stopped at the user's request during epoch 4
- Authorization: user requested this run after the matched 1000-update
  WikiText-103 experiment completes
- Upstream protocol producer:
  `../../../train_wikitext2_selective_innovation_20epoch_17m.py`
- Current-architecture source run:
  `../EXP-20260726-k4-ha-skew-stride1-batch64-micro16-state-conditioned-givens-latent-tf-open-nll-ema-warmstart100-transition-only-closure-13m/`
- Test split is authorized only once, after all 20 epochs, using the
  validation-selected checkpoint

### Execution update

- Complete validation/checkpoint rows are preserved through epoch 3,
  update 876.
- The process was interrupted during an epoch-4 backward pass to inspect
  sentence quality. No partial epoch-4 state is treated as a checkpoint.
- Epoch-3 validation H1 Open PPL is `312.653338`.
- The test split was not read.

## Question

Under the existing size-matched WikiText-2 20-epoch protocol, how well does
the single deterministic state-conditioned latent transition learn ordinary
next-token likelihood and self-fed H2--H4 behavior when behavioral closure is
distilled from a warm-started full-model EMA teacher?

## Fixed architecture

The central transition is:

`T(h) = K(h)h + R(h)`.

- GPT-2 vocabulary 50,257
- width 336 and two reversible causal encoder blocks
- exact inverse decoder and RMS-tied vocabulary head
- two-stage state-conditioned exact Givens operator
- state-conditioned deterministic innovation
- transition bottleneck 128
- 17,696,112 trainable parameters
- canonical latent teacher-forced one-step inputs
- one sequential self-fed latent Open rollout
- no discrete-token or embedding feedback in the central recurrence
- no sampled noise, candidate paths, path prior, trajectory comparison,
  latent plan, or latent regression loss

Each 256-token chunk uses prefix length 252, stride-one anchors 0 through 251,
and horizons B through E.

## Loss

For each actual-prefix canonical state:

`L_TF = mean_{B:E} CE(logits_TF,j, y_j)`.

For one self-fed rollout from each canonical root:

`L_OL = mean_{C:E} CE(logits_OL,j, y_j)`.

The complete gradient-free EMA model supplies canonical teacher
distributions:

`L_close = mean_{C:E} KL(p_EMA,TF,j || p_online,OL,j)`.

At optimizer update `s`:

`L = L_TF + L_OL + min(1, s/100) L_close`.

- TF and Open NLL update the complete online model.
- Closure gradients update only the online central transition.
- The EMA receives no gradients or optimizer state.
- H1 is excluded from Open NLL and closure because TF and Open execute the
  same first transition.

## EMA schedule

- after updates 1 through 100: exact copy, decay 0
- after update 101 onward: decay 0.99
- during the exact-copy phase the online canonical logits are used directly,
  which is mathematically identical and avoids a redundant full EMA forward
- validation always evaluates the explicit EMA teacher
- both EMA-target closure and current-online canonical closure are reported

## External WikiText-2 protocol

- Hugging Face `wikitext/wikitext-2-raw-v1`
- `GPT2Tokenizer.from_pretrained("gpt2")`
- remove empty rows, join with newline, tokenize each complete split, and
  truncate to non-overlapping 256-token chunks
- registered chunks: train 9,343; validation 965; test 1,106
- seed 1337 for model initialization and train-chunk shuffle
- effective batch 32; physical microbatch 4 with eight exact accumulations
- 20 complete epochs, 292 optimizer updates per epoch, 5,840 total updates
- AdamW betas `(0.9, 0.95)`, weight decay 0.01
- initial LR `3e-4`; per-update cosine annealing over all 5,840 updates
- gradient clip 1.0
- strict float32; TF32 disabled
- full validation after every epoch
- online and EMA state saved together
- metrics in TSV; interpretation in a separate Markdown file

Registered train/validation tokenized-data provenance:

| split | raw rows | nonempty rows | untruncated tokens | stored tokens | chunks | token int32 SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| train | 36,718 | 23,767 | 2,391,884 | 2,391,808 | 9,343 | `ba036632a98cebca20c2cb678d3547b7f607f4cb8e4e540814bcba18bd44af89` |
| validation | 3,760 | 2,461 | 247,289 | 247,040 | 965 | `e2663296a2cd2e1d6b72deb89afb76102d0dffbf2d6357af9042cb700594d02c` |

The test checksum and provenance are recorded only with the authorized final
evaluation, rather than reading that split during preregistration.

### Pre-run CUDA capacity update

Before optimizer update 1, physical microbatch 4 passed the complete
EMA-teacher backward and fused-AdamW step at `9,358,753,280` peak allocated
bytes. An isolated physical-microbatch-8 smoke also passed at
`18,487,671,808` bytes, but the complete producer process subsequently failed
before update 1: after full structural preflight and DataLoader setup,
`log_softmax` could not allocate another `1.13 GiB` with only `956 MiB`
available. The zero-row evidence is preserved as
`failed_microbatch8_run.tsv` and `failed_microbatch8_metrics.tsv`.

The production run therefore uses the verified physical microbatch 4 with
eight exact accumulations. Effective batch, sample order, optimizer updates,
LR schedule, and all loss weights remain unchanged.

## Primary metric

H1 Open and canonical TF are the same target-free next-token prediction. The
ordinary comparable validation metric is therefore:

`PPL_H1 = exp(mean CE(logits_OL,H1, y_H1))`.

The best checkpoint is selected only by validation H1 Open NLL. H2--H4 Open
NLL, closure, agreement, latent relative MSE, angle, and innovation scale are
diagnostics and do not select the checkpoint.

Published size-matched references retained from the existing manifest are:

- Grassmann v4: validation PPL 236.05, test PPL 242.94
- Transformer: validation PPL 190.03, test PPL 198.17

## Preflight and success criteria

Before update 1:

- parameter count is exactly 17,696,112
- online and EMA parameters/logits are exact copies
- the transition is identity initialized
- H1 teacher/Open states are identical
- Open states and logits do not depend on the held-out last four tokens
- online transition and encoder gradients are finite and nonzero
- EMA parameters have no gradients
- all train/validation checksums and chunk counts match the registered data

Run success:

- complete all 20 epochs without a non-finite metric
- best validation H1 PPL at most 236.05 is the primary external target
- best validation H1 PPL at most 190.03 is the stretch target
- report whether online H2--H4 closure and top-1 agreement improve across
  epochs without hiding any Open-NLL deterioration
- read the test split exactly once after selecting the best validation epoch

## Producer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_wikitext2_state_conditioned_latent_tf_ema_20epoch_17m.py \
  --epochs 20 --batch 32 --microbatch 4 --eval-microbatch 1 --workers 4 \
  --record-dir \
    experiments/records/EXP-20260726-wikitext2-gpt2tok-20epoch-state-conditioned-givens-latent-tf-open-nll-ema-warmstart100-transition-only-closure-17m \
  --output-dir \
    outputs/experiments/EXP-20260726-wikitext2-gpt2tok-20epoch-state-conditioned-givens-latent-tf-open-nll-ema-warmstart100-transition-only-closure-17m
```

## Evidence boundary

EMA closure is a behavioral regularizer; it does not introduce stochastic
branch sampling or prove exact sequence-level distribution equivalence.
WikiText-103 and WikiText-2 results use different tokenizers and parameter
allocations and are not compared as raw NLL values.
