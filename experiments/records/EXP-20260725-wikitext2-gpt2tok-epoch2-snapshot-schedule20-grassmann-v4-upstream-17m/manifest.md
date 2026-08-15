# EXP-20260725 Grassmann v4 epoch-2 snapshot on 20-epoch schedule

## Status and provenance

- State: preregistered before optimization update 1
- Authorization: correction required for the user-requested matched-epoch
  sentence-quality comparison
- Repository commit: `67efbc158ad823f7196f0696415f6e32b5e2e2fa`
- Producer: unmodified upstream `train_wikitext2.py`
- Launch uses `--epochs 20`; execution is stopped only after epoch 2 has
  completed and its validation-selected checkpoint has been written
- Seed: upstream-unset; `PYTHONHASHSEED=1337` only

### Execution result

- Epoch 1: train NLL `7.2350`, validation NLL `6.4294`, PPL `619.77`
- Epoch 2: train NLL `6.1279`, validation NLL `6.0710`, PPL `433.11`
- The process ended after the epoch-2 best checkpoint was written and before
  an epoch-3 validation result. The test split was not evaluated.
- Sentence evidence is recorded under `epoch2_sentence_comparison/`.

## Question

Produce a Grassmann epoch-2 checkpoint whose optimizer and cosine scheduler
are at update 584 of 5,840, matching the latent epoch-2 checkpoint's schedule
fraction.

## Fixed protocol

- same WikiText-2/GPT-2-tokenizer 256-token chunks
- Grassmann v4 width 256, six layers, 17,695,168 parameters
- batch 32, AdamW LR `3e-4`, weight decay `0.01`
- `CosineAnnealingLR(T_max=5,840)`
- stop after the epoch-2 checkpoint is durably present; do not evaluate test
- use this checkpoint for Grassmann greedy AR versus latent prior block-4
  generation on the eight prompts preregistered in the sibling comparison
  manifest
