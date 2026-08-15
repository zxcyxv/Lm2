# EXP-20260731 K=1 CE with attached self-redecode KL T

## Status

- State: preregistered before optimizer update 1
- Authorization: user requested the no-stop-gradient counterpart after the
  stop-gradient run and its alternating K--T audit completed
- Primary paired control:
  `EXP-20260731-k1-ce-self-canonical-redecode-kl-sg-13m`
- Test split remains unread.

## Question

Does allowing the self-reencoded teacher state and its immediate redecode
distribution to co-adapt with the student stabilize the behavioral collapse
operator and reduce the severe CE optimization conflict observed when that
teacher was stopped?

## Architecture

The model and operator contracts are identical to the paired control:

```text
K_A = MLP_K(h_A)
q_B = K_A h_A

T_B = MLP_T(q_B)
hhat_B = T_B q_B
```

K and T have distinct width-128 controller MLPs. K is a
prefix/state-conditioned normal spectral operator. T is an
identity-initialized proposal-conditioned spectral operator in the same
learned orthogonal basis. The backbone is the same width-896, two-block
reversible causal encoder with analytic exact-inverse decoder and RMS-tied
head.

## Objective and the sole ablation

The ordinary language path remains:

```text
p_B  = softmax(Head(Decode(prefix_A, q_B)))
L_CE = CE(p_B, gold_(A+1))
```

T is absent from this path. The model's own greedy action is
`b_B = argmax(p_B)`. Argmax remains non-differentiable, but every operation
after that discrete selection is attached:

```text
hstar_B = Encode(prefix_A, b_B)
r_B     = softmax(Head(Decode(prefix_A, hstar_B)))
s_B     = softmax(Head(Decode(prefix_A, T_B q_B)))

L_meta = KL(r_B || s_B)
L      = L_CE + L_meta
```

Unlike the paired control, `hstar_B` and `r_B` are not detached. KL gradients
flow through both teacher and student branches into their shared reversible
E/D and tied head. The student branch additionally updates T and q/K. The
selected token index itself carries no gradient.

The native near-one-hot teacher distribution is used without temperature.
There is no latent MSE, horizon CE, gold-token canonical target, EMA teacher,
branch loss, shell projection, or T in the CE path.

## Intended inference

Inference is unchanged and alternates the two input-conditioned operators:

```text
q_B = K(h_A) h_A
h_B = T(q_B) q_B
q_C = K(h_B) h_B
h_C = T(q_C) q_C
...
```

The corrected tape is decoded jointly. A step-1000 block-4 audit will use the
same preserved validation starts and compare against raw greedy AR with
literal selected-token re-encoding.

## Fixed configuration

- WikiText-103 train and validation; BPE vocabulary 8192
- seed 1337; validation starts use seed `1337 + 999`
- test split unread
- width 896; two reversible causal blocks
- exact inverse causal decoder and RMS-tied head
- alpha-zero prefix-conditioned normal spectral K
- K and T controller bottlenecks 128
- T maximum absolute log-scale 2.0 and phase correction pi
- T identity initialization
- one-token horizon; stride-one anchors 0 through 255
- effective batch 64; physical microbatch 16; four accumulations
- one AdamW optimizer; global clip norm 1.0
- KL weight 1.0; native temperature 1.0
- 1000 updates on the unchanged 6000-update LR schedule
- validation examples 128
- reports at steps 1, 50, 100, 250, 500, 750, and 1000
- checkpoints at steps 100, 250, 500, 750, and 1000
- strict float32; TF32 disabled

## Structural criteria

- K and T use distinct controller MLPs and consume `h_A` and `q_B`,
  respectively.
- T is identity at initialization below relative error `1e-6`.
- changing the future corpus token changes none of the self-selected paths.
- canonical self-action encoding matches literal prefix-plus-action encoding
  below `5e-5`.
- canonical teacher states and teacher logits require gradients.
- CE gives zero or absent gradient to every T parameter.
- meta KL gives finite nonzero gradients to T, q/K, the teacher branch, the
  student decoder branch, and the tied embedding/head.
- all values and gradients remain finite.

## Step-1000 criteria

Use the same paired thresholds as the stopped control:

- validation raw-q NLL falls below its step-1 value;
- corrected KL is lower than raw-q teacher KL, with ratio at most `0.50`;
- corrected/teacher top-1 agreement is at least `0.90`;
- raw-AR/alternating-K--T token agreement is at least `0.90`;
- exact block-4 agreement is at least `0.75`.

Additionally compare NLL, KL stability, total gradient norm, throughput, and
peak VRAM directly with the paired stopped-teacher record. Metrics are TSV,
interpretation is Markdown, and checkpoints remain outside Git.
