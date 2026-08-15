# EXP-20260725 WikiText-2 upstream Grassmann v4, 2 epochs

## Status and provenance

- State: preregistered before optimization update 1
- Authorization: user requested a fresh two-epoch Grassmann run for a
  matched-epoch sentence-quality comparison
- Repository: `https://github.com/Infatoshi/grassmann-flows`
- Commit: `67efbc158ad823f7196f0696415f6e32b5e2e2fa`
- Producer: unmodified upstream `train_wikitext2.py`
- Model source: unmodified upstream `src/models/grassmann_v4.py`
- Seed: upstream producer does not set one; `PYTHONHASHSEED=1337` is recorded
  but does not make PyTorch initialization or shuffling deterministic

### Execution result and scope correction

- The run completed two epochs with validation PPL `679.22` and `594.82`.
- This run is not used for the matched-epoch generation comparison because
  upstream couples `epochs` to `CosineAnnealingLR.T_max`. Setting
  `--epochs 2` annealed LR to zero by update 584, while the latent epoch-2
  checkpoint used the first 584 updates of a 5,840-update schedule.
- The checkpoint and raw evidence remain preserved as the distinct
  two-epoch-budget/two-epoch-schedule condition.

## Question

At the same two-epoch budget, how does upstream Grassmann v4 greedy AR
sentence quality compare with the width-256/depth-24 latent-orbit model's
target-free block-4 generation?

## Fixed training protocol

- GPT-2 tokenizer and vocabulary 50,257
- `wikitext/wikitext-2-raw-v1`
- discard empty rows, newline join, non-overlapping 256-token chunks
- width 256, six Grassmann v4 blocks, 17,695,168 parameters
- batch 32, two epochs
- upstream AdamW at LR `3e-4`, weight decay `0.01`
- cosine annealing over 584 updates and gradient clipping 1.0
- full validation after each epoch

## Generation comparison

- latent checkpoint: width-256/depth-24 epoch 2 `best.pt`
- Grassmann checkpoint: validation-selected best checkpoint through epoch 2
- eight fixed validation chunks selected by seed 1337
- first 64 tokens are the prompt and next 64 are the reference
- Grassmann: greedy AR, one token per re-encoding step
- latent model: target-free prior-argmax block 4, four tokens per
  re-encoding step, innovation noise seed 3361
- token repetition/diversity metrics are TSV; decoded evidence and
  interpretation are Markdown

The generation policies are architecture-native and intentionally asymmetric:
Grassmann has no native block-4 decoder.

## Success and evidence boundary

- complete two epochs without non-finite loss
- preserve the upstream raw log and best checkpoint
- compare repetition, diversity, reference continuation accuracy, and decoded
  text; eight prompts are qualitative evidence and not a population estimate
