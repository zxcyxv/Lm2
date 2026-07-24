# EXP-20260721: K=1 decoder inverse-constraint ablation at 13M

## Question

When the one-step latent operator `K` is trained only by exact teacher-forced
next-token cross entropy, how much does requiring the decoder to be the exact
analytic inverse of the encoder constrain performance?

## Fixed prediction contract

For every gold anchor token `x_t`, all variants compute an encoder tape from
the gold prefix, apply one shared trainable bias-free linear operator to the
last causal state, append that state to the same prefix tape, and predict
`x_(t+1)`. Branches belonging to distinct anchors cannot attend to one another.

The sole optimized objective is next-token cross entropy over every anchor.
There is no hidden-state MSE, closure loss, JEPA target, central planner,
multi-step loss, spectral constraint, straight-through estimator, or generated
token reinsertion.

## Comparisons

| Variant | Decoder | Token head | Width | Distinct encoder/decoder blocks | Target parameters |
|---|---|---|---:|---:|---:|
| `inverse_cosine` | analytic inverse of encoder | tied cosine codebook | 896 | 2 shared reversible blocks | about 13M |
| `independent_cosine` | separately learned causal stack | tied cosine codebook | 736 | 2 encoder + 2 decoder blocks | about 13M |
| `inverse_rms` | analytic inverse of encoder | RMSNorm + tied dot-product | 896 | 2 shared reversible blocks | about 13M |

Widths differ because an exact inverse reuses encoder parameters whereas an
independent decoder owns a second stack. This is a matched-total-parameter
comparison, not a matched-width causal identification experiment.

## Data and optimization

- Dataset: local WikiText-103 BPE binary manifest, train and validation only.
- Vocabulary: 8192.
- Seed: 1337 for initialization, training windows, and fixed validation starts.
- Context: 256 tokens; every position predicts the following gold token.
- Batch: 128 windows (32,768 target tokens per optimizer step).
- Steps: 1000 per variant, same schedule and sampled training starts.
- Precision: FP32 tensors with TF32 CUDA matrix multiplication enabled; no autocast.
- Optimizer: AdamW, beta `(0.9, 0.95)`, zero weight decay, gradient clip 1.0.
- Checkpoint selection: lowest validation NLL among preregistered evaluation
  steps. The test split is not consulted.

## Primary criterion

At matched parameter budget and training tokens, compare the full validation
NLL curves and the best validation NLL. A lower independent-decoder NLL is
evidence that the exact-inverse constraint creates an optimization or
representation burden in this experimental scope. It does not by itself
identify which mechanism causes the gap.

## Secondary diagnostics

- next-token accuracy;
- relative MSE and cosine between `K h_t` and the encoder state obtained after
  appending the gold next token (measurement only, never a loss);
- gradient norm received by `K`;
- codebook row self-retrieval rate under the active token head;
- throughput, wall time, and peak allocated VRAM.

## Success and stopping

All three runs must pass finite-logit and nonzero-`K`-gradient preflight,
complete 1000 optimizer steps, and use the same fixed validation windows.
Intermediate best points are not treated as test results. If one variant hits
a numerical failure, preserve the trace and report it rather than silently
changing its precision or objective.
