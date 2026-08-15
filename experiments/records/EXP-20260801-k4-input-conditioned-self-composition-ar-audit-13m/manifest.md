# EXP-20260801 input-conditioned self-composition AR audit

## Status

- State: preregistered before the corrected checkpoint evaluation
- Authorization: user requested rechecking the trained checkpoint with each
  continuous output fed back into the input-dependent K generator
- Test split remains unread.

## Correction being tested

Let `G(h)` generate a matrix and define the transition

`F(h) = G(h) h`.

The intended continuous parallel sequence is

`u_0 = h_t`

`u_(j+1) = F(u_j) = G(u_j) u_j`.

Consequently

`u_1 = K_t h_t`

`u_2 = K_(t+1) K_t h_t`, where `K_(t+1) = G(K_t h_t)`.

The previous evaluator incorrectly used `G(h_t)^j h_t`, freezing the first
matrix. This audit regenerates the matrix after every continuous step.

## Reference AR sequence

At each shared literal prefix boundary, the AR path:

1. applies `G(c_(j-1)) c_(j-1)`;
2. exact-inverse decodes it and takes tied-codebook greedy argmax;
3. appends that selected token to its own history; and
4. canonically re-encodes the generated history to obtain `c_j`.

No held-out future token is inserted into `c_j`.

## Decomposed metrics

For horizons one through four, report separately:

- overall equivalence: `u_j` versus canonical greedy-AR state `c_j`;
- one-step closure: `G(c_(j-1))c_(j-1)` versus `c_j`;
- accumulated path difference: `u_j` versus
  `G(c_(j-1))c_(j-1)`;
- continuous-composition versus greedy-AR token agreement.

This decomposition prevents the one-step codebook closure error from being
confused with error accumulated by continuous self-composition.

## Fixed evidence and split

- producer experiment:
  `EXP-20260801-k4-input-conditioned-kpower-h1-ce-only-tied-ar-closure-13m`
- fresh deterministic step-0 initialization plus producer checkpoints at
  steps 500 and 1000
- step-500 checkpoint SHA-256:
  `05f2f478e1a879707ce04dc871c0fe74856f5ece00fa56d9c9cad14f632bab30`
- step-1000 checkpoint SHA-256:
  `0fb84a5393525d922754145cd0644ba49ac5331250a7449d427221c149fc59c1`
- exact producer validation starts, seed `1337 + 999`, 64 examples
- WikiText-103 BPE-8192 validation split; test split unread
- context 256, horizon four, boundary stride four
- strict float32 and greedy selection

## Criterion

Relative to the deterministic step-0 model, the corrected self-composition
counts as improved only if step 1000 has higher overall state cosine and lower
overall relative MSE at at least three of four horizons. Token agreement is
reported but is not allowed to substitute for state agreement.

Numeric results are TSV and interpretation is Markdown. The original frozen-K
metrics remain preserved with their corrected scope.
