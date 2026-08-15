# EXP-20260725 K=4 selective innovation tape, 6000 updates

## Status

- State: preregistered before optimization update 1
- Authorization: user-requested fresh 6000-update training run
- Parent architecture:
  `../EXP-20260725-k4-ha-skew-stride1-batch64-micro16-selective-innovation-tape-three-trajectory-prior-tau005-h1-online-mse-13m/`
- This is a fresh seed-1337 run, not a resume from the step-1000 checkpoint.
- Test split remains unread.

## Question

Does training the unchanged selective-innovation architecture for the full
6000-update learning-rate schedule improve one-step token accuracy,
validation marginal NLL, and generated-text coherence without collapsing
trajectory diversity?

The model, forward equations, and objective are identical to the parent
manifest:

`u_i,0 = hA`,

`u_i,j+1 = K u_i,j + r_i,j+1`,

`L = L_path + relMSE(K hA, hB_online) + L_prior`.

No new model component or loss is introduced.

## Fixed training configuration

- WikiText-103 train and validation splits; BPE vocabulary 8192
- seed 1337
- width 896, two reversible causal blocks, 13M parameters
- exact inverse decoder and rms-tied head
- global `K=exp(A-A^T)`
- three sampled trajectories, four horizons, posterior temperature 0.05
- clean attached online h1 relative-MSE weight 1
- prior KL weight 1
- effective batch 64, microbatch 16, four accumulations
- strict float32, TF32 disabled
- AdamW and clip norm 1.0
- 6000 fresh optimizer updates on the 6000-update LR schedule
- validation starts: 128 examples from seed `1337+999`
- validation reports through step 1000 as in the parent, then every 500 steps
- persistent checkpoints at steps 1000, 2000, 3000, 4000, 5000, and 6000

## Step-3000 comparison

At step 3000, compare against the preserved standard GPT-2 AR checkpoint
trained for 1000 updates in the preceding `/workspace/Lm` project.

- GPT-2 producer:
  `/workspace/Lm/train_gpt2_baseline_matched_13m.py`
  (SHA-256
  `a7dcd2e590a561399e238ded898b40376e8e84735781ad415fa27d70d41edd0a`)
- GPT-2 checkpoint:
  `/workspace/Lm/outputs/experiments/EXP-20260723-gpt2-baseline-matched-13m/best.pt`
  (step 1000, SHA-256
  `bfff4c3f369b034fd979bcec564a1a136c869fadc6c36b89f34c8a7d68963522`)
- GPT-2 architecture: standard pre-LayerNorm GPT-2, width 440, four layers,
  eight heads, 13,033,680 parameters, teacher-forced next-token CE only.
- Use the same validation split and tokenizer; do not read the test split.
- Quantitative fields: h1 token NLL and top-1 accuracy. For the trajectory
  model, report path/posterior/prior policy variants rather than silently
  choosing the best one.
- Qualitative fields: fixed validation-derived prefixes, fixed decoding
  settings, generated continuations, repetition statistics, and an explicit
  note that the trajectory model uses block-4 generation while GPT-2 uses AR.
- Text samples and quantitative metrics are separate artifacts; interpretation
  is written to Markdown, not into the metrics TSV.

## Success and failure criteria

- No non-finite optimization or validation metric through step 6000.
- Step-3000 validation marginal NLL and posterior h1 accuracy improve over
  this architecture's fresh step-1000 values.
- Clean h1 relative MSE remains at most 0.03.
- Innovation scale remains finite and nonzero, and trajectory diversity is
  finite at every horizon.
- GPT-2 comparison is descriptive evidence, not a pass/fail gate, because its
  training objective and generation policy differ.

## Evidence boundary

The run tests optimization depth for one fixed architecture. It does not by
itself establish long-context quality, wall-clock generation speed, or
equivalence to autoregressive maximum likelihood.

## Registered step-4000 block-length diagnostic

Before executing the user-requested block sweep:

- checkpoint: `step4000.pt`, SHA-256
  `f7b27f871e469aaadd3681ae76bfb93afb359778ef011291e5f25576dc3f365c`
- split: validation only; test remains unread
- examples: the same 64 starts generated from seed `1337+2024` used by the
  registered GPT-2 comparison
- prompt and continuation length: 64 and 64 tokens
- policy: sampled-candidate prior selection followed by greedy token decoding
- fixed evaluation noise seed: `1337+2024`, reset independently for each
  block-length policy
- block lengths: 2, 4, 8, 16, and 32
- control: the preserved step-1000 GPT-2 greedy AR continuation
- metrics: reference accuracy, distinct-2/4, immediate repetition,
  period-block repetition, exact adjacent-block repetition, repeated 4-gram
  coverage, collapse frequency, and longest identical run
- success criterion: descriptive scaling curve; quality should not show a
  sharp collapse at block 8 or above if the learned scan and global operator
  extrapolate beyond their trained horizon

Only block 4 lies at the trained horizon. Block 2 is interpolation and blocks
8, 16, and 32 are untrained-horizon extrapolations.
