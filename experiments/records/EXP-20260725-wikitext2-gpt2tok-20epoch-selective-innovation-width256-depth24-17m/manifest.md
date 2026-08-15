# EXP-20260725 WikiText-2 selective innovation, width 256/depth 24

## Status

- State: preregistered before optimization update 1
- Authorization: user requested replacing the width-336/depth-2 run with a
  width-256 model whose depth makes total parameters as close as possible to
  upstream Grassmann v4
- Producer:
  `train_wikitext2_selective_innovation_width256_depth24_20epoch_17m.py`
- Seed: 1337
- Test split is authorized only for one final evaluation of the
  validation-selected best checkpoint

### Execution update

- The run was stopped at the user's request after epoch 2 completed and while
  epoch 3 was in progress.
- Complete validation rows through epoch 2 and the epoch-2 `best.pt`/`last.pt`
  checkpoints are preserved.
- No test-split evaluation was performed; this is not a completed 20-epoch
  result.

## Question and controls

Does reallocating parameters from the GPT-2 vocabulary embedding into the
reversible causal backbone materially close the WikiText-2 target-free
perplexity gap?

Controls:

- width-336/depth-2 selective-innovation run, stopped after epoch 8:
  validation h1 mixture PPL `285.34`
- local upstream Grassmann v4 run, stopped after epoch 13:
  best validation PPL `238.57`
- published upstream Grassmann target: best validation PPL `236.05`

## Parameter matching

| model | total | embedding/position | non-embedding |
|---|---:|---:|---:|
| this experiment | 17,715,074 | 12,865,792 | 4,849,282 |
| upstream Grassmann v4 | 17,695,168 | 12,931,328 | 4,763,840 |

The total difference is 19,906 parameters (`+0.11%`), and the
non-embedding difference is 85,442 (`+1.79%`). Width 256 with 24 reversible
blocks is the nearest integer-depth match using the unchanged block
definition. Width 256 with 23 blocks has 17,517,570 parameters and is farther
from the target.

## Architecture and loss

- GPT-2 vocabulary 50,257
- width 256
- 24 causal reversible additive-coupling blocks
- exact inverse decoder and rms-tied output head
- global orthogonal `K=exp(A-A^T)`
- three trajectories and four trained horizons
- selective innovation state width 64 and noise width 32

The forward, candidate prior, and objective are unchanged from the
width-336/depth-2 control:

`u_i,0 = hA`,

`u_i,j+1 = K u_i,j + r_i,j+1`,

`L = L_path + relMSE(K hA, hB_online) + KL(q || pi)`.

No new loss term, normalization, or geometry constraint is introduced.

## Data and optimization

- Hugging Face `wikitext/wikitext-2-raw-v1`
- GPT-2 tokenizer, nonempty-line newline join
- non-overlapping chunks of 256 tokens
- expected chunks: train 9,343; validation 965; test 1,106
- prefix length 252 and horizons 1 through 4
- 20 epochs
- effective batch 32
- physical microbatch 4 and exact gradient accumulation 8. The preregistered
  microbatch-8 no-update backward succeeded but allocated 31,322,263,040
  bytes, leaving insufficient optimizer/allocator margin. The permitted
  fallback microbatch 4 passed a real accumulated backward, clip, and fused
  AdamW step with peak allocation 15,781,262,848 bytes before update 1 of the
  recorded run
- AdamW betas `(0.9, 0.95)`, weight decay `0.01`
- LR `3e-4`, per-update cosine annealing, gradient clip 1.0
- strict float32 with TF32 disabled
- full validation after each epoch

## Metrics and success

- primary metric: target-free h1 candidate-mixture validation NLL/PPL over
  252 anchors
- diagnostics: prior/confidence/oracle h1 metrics, prior winner agreement,
  innovation scale/diversity, and clean/branch relative MSE
- no non-finite metric
- clean h1 relative MSE at most 0.03
- first matched checkpoint: epoch-8 PPL below `285.34`
- primary: best validation PPL at most `236.05`
- stretch: best validation PPL at most `190.03`
- metrics are TSV; interpretation remains in Markdown
