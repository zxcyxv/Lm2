# EXP-20260725 WikiText-2 selective innovation, 20 epochs

## Status

- State: preregistered before optimization update 1
- Authorization: user-requested WikiText-2 reproduction-style training
- Model order: train the latent-orbit model first; local Grassmann and
  Transformer reruns are deferred
- Test split is authorized only for one final evaluation of the
  validation-selected best checkpoint

### Execution update

- The run was stopped at the user's request after epoch 9 had completed and
  while epoch 10 was in progress.
- `metrics.tsv` therefore contains complete validation rows through epoch 9.
  No test-split evaluation was performed, and the run is not treated as a
  completed 20-epoch result.

## External comparison protocol

- Repository: `https://github.com/Infatoshi/grassmann-flows`
- Commit: `67efbc158ad823f7196f0696415f6e32b5e2e2fa`
- Producer: `train_wikitext2.py`
- Dataset: Hugging Face `wikitext/wikitext-2-raw-v1`
- Tokenizer: `GPT2Tokenizer.from_pretrained("gpt2")`, vocabulary 50,257
- Preprocessing: remove empty lines, join with newline, tokenize the complete
  split, truncate to non-overlapping 256-token chunks
- Registered external chunk counts: train 9,343, validation 965, test 1,106
- External training protocol: batch 32, 20 epochs, AdamW, peak LR `3e-4`,
  weight decay `0.01`, per-update cosine annealing, gradient clip 1.0
- Published reproduction results:
  Grassmann v4 17,695,168 parameters, best validation PPL 236.05 and test PPL
  242.94; Transformer 17,670,400 parameters, best validation PPL 190.03 and
  test PPL 198.17

## Question

Under the same WikiText-2 tokenization, chunking, optimizer, token batch, and
20-epoch budget, can the selective latent-orbit model reach or exceed the
published Grassmann reproduction and size-matched Transformer while retaining
usable target-free block-4 generation?

## Model

- vocabulary 50,257
- width 336, two reversible causal encoder blocks
- exact inverse decoder and rms-tied output head
- global orthogonal `K=exp(A-A^T)`
- three sampled trajectories, four trained horizons
- selective innovation state width 64 and noise width 32
- approximately 17.73M trainable parameters
- no dropout is added to the native latent-orbit architecture

Each 256-token chunk uses prefix length 252 and horizons 1 through 4:

`u_i,0 = hA`,

`u_i,j+1 = K u_i,j + r_i,j+1`,

`L = L_path + relMSE(K hA, hB_online) + KL(q || pi)`.

The MSE is attached and applies only to the clean first transition. There is
no h2--h4 state regression and no additional objective.

## Fixed training configuration

- seed 1337 for model initialization, chunk shuffling, and sampled noise
- effective batch 32; physical microbatch 8 with four exact gradient
  accumulations
- 20 complete passes over the 9,343 train chunks
- AdamW betas `(0.9, 0.95)`, weight decay `0.01`
- initial LR `3e-4`, `CosineAnnealingLR` over all optimizer updates
- gradient clipping 1.0
- strict float32 with TF32 disabled
- full validation split after every epoch
- best checkpoint selected by proper target-free h1 mixture NLL
- metrics in TSV; interpretation and generation samples in Markdown

The optimizer-free CUDA capacity smoke used random `[8,256]` GPT-2-vocabulary
tokens, passed the complete objective backward, observed finite nonzero
global-K and innovation gradients, and peaked at 21,039,805,440 allocated
bytes on the RTX 5090.

## Comparable language-model metric

The low-temperature four-token training objective is not reported as ordinary
perplexity. For comparison with the external AR models, h1 predictive
probability is the target-free candidate mixture

`p(y | prefix) = sum_i pi_i p_i(y | prefix)`.

Validation and test NLL use

`-log sum_i pi_i p_i(y_gold | prefix)`,

over anchors 0 through 251. Top-1 accuracy uses the argmax of the same mixture.
Prior-argmax, confidence-selected, path-mean, and label-informed oracle metrics
are diagnostics and are reported separately.

## Sentence-quality audit

After training, the best-validation checkpoint generates 64 tokens from 64
fixed 64-token validation prefixes.

- target-free sampled-candidate prior
- greedy token decoding
- block 4 primary, block 8 diagnostic
- evaluation start/noise seed `1337+2024`
- reference accuracy, distinct n-grams, immediate/periodic/exact-block
  repetition, repeated 4-gram coverage, collapse rate, and decoded samples

## Success criteria

- no non-finite training or validation metric
- clean h1 relative MSE at most 0.03
- finite nonzero innovation scale and finite diversity at every horizon
- primary: best proper h1 validation PPL at most 236.05, matching the
  published Grassmann reproduction
- stretch: best validation PPL at most 190.03, matching its Transformer
- final test PPL is read exactly once from the validation-selected checkpoint
- sentence evidence is reported independently of PPL; low PPL alone does not
  establish coherent block generation

## Evidence boundary

The external numbers are published reference points until their checkpoints
are trained locally under the fixed seed. Architecture-native losses and
decoding policies differ, so only the registered proper h1 mixture NLL/PPL is
treated as a likelihood comparison.
