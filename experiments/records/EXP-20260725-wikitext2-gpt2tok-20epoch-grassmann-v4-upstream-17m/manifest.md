# EXP-20260725 WikiText-2 upstream Grassmann v4, 20 epochs

## Status and provenance

- State: preregistered before optimization update 1
- Authorization: user requested stopping the active latent-orbit run and
  training Grassmann Flow through the same WikiText-2 comparison protocol
- Repository: `https://github.com/Infatoshi/grassmann-flows`
- Commit: `67efbc158ad823f7196f0696415f6e32b5e2e2fa`
- Producer: the unmodified upstream `train_wikitext2.py`
- Model source: the unmodified upstream `src/models/grassmann_v4.py`
- Seed: upstream producer does not expose or set a seed; the launch process
  sets `PYTHONHASHSEED=1337`, but PyTorch initialization and DataLoader
  shuffling remain upstream-unseeded behavior
- Test split is read once after 20 epochs using the
  validation-selected best checkpoint, as implemented upstream

### Execution update

- The upstream run was stopped at the user's request after epoch 13 completed.
- Best observed validation loss through epoch 13 was `5.4747` at epoch 12
  (PPL `238.57`); epoch 13 was `5.4749` (PPL `238.63`).
- The upstream best checkpoint and raw stdout log are preserved. No test-split
  evaluation was performed, so this is not a completed 20-epoch result.

## Question

Can the upstream Grassmann v4 implementation locally reproduce its published
WikiText-2 validation and test perplexity curve on this RTX 5090 environment,
and how does its matched-epoch validation curve compare with the interrupted
latent-orbit run?

## Model and forward

- `GrassmannGPTv4`
- vocabulary 50,257
- width 256, six blocks, FFN width 1,024
- reduced Grassmann dimension 32
- causal window offsets `[1, 2, 4, 8, 12, 16]`
- dropout 0.1
- tied token embedding and LM head
- 17,695,168 trainable parameters

The loss is ordinary next-token autoregressive cross entropy over positions
1 through 255. No latent-orbit, trajectory, MSE, or auxiliary loss is added.

## Data and optimization

- Hugging Face `wikitext/wikitext-2-raw-v1`
- `GPT2Tokenizer.from_pretrained("gpt2")`
- discard empty rows, join remaining rows with newline, tokenize once per
  split, and truncate to non-overlapping 256-token chunks
- expected chunks: train 9,343; validation 965; test 1,106
- batch 32, no gradient accumulation
- 20 epochs
- upstream `AdamW` defaults except LR `3e-4` and weight decay `0.01`
- `CosineAnnealingLR` over all updates
- gradient clipping 1.0
- upstream float32 CUDA execution
- full validation after every epoch

## Metrics and comparison boundary

The outer language-model metric is the standard token-weighted
`exp(mean cross entropy)`. All chunks contain the same 255 supervised
positions, so the upstream example-mean implementation equals a token mean.

The upstream stdout and final JSON are primary raw evidence. Epoch metrics are
transcribed without reinterpretation to `metrics.tsv`; conclusions are kept
in a separate Markdown file. The latent-orbit run uses 252 h1 anchors and a
sampled three-candidate prior mixture, so matched-epoch PPL is an informative
benchmark rather than an identical predictive distribution.

## Success criteria

- no non-finite loss or CUDA failure
- complete all 20 epochs
- locally measured best validation PPL at most 236.05
- final test PPL at most 242.94
- report parameter count, epoch time, and peak allocated CUDA memory
