# Step-100 same-operator AR versus block-4 audit

## Status

- State: preregistered before generation
- Authorization: user-requested intermediate generation comparison
- Checkpoint step: 100
- Checkpoint SHA-256:
  `cf06bf5056cf5fbf1cef38d510820654cea74a6958f3484951594f60d7857006`
- Test split remains unread

## Question

At the first saved operator-commitment checkpoint, are AR and block-4
generation already approximately equivalent when both retain exactly the
same branch operator within each four-token block?

This audit distinguishes:

1. **same-prefix local agreement**, which starts both policies from the same
   validation prefix and compares one four-token continuation; and
2. **free-running agreement**, where each policy generates 64 tokens and may
   acquire a different prefix after its first disagreement.

## Matched policies

At each four-token block boundary, both policies encode their current prefix,
select the prior-argmax branch once, and compile one
`T_i = Q_i K`.

- `same_operator_block4` applies
  `T_i hA, T_i^2 hA, T_i^3 hA, T_i^4 hA`, decodes the four states jointly,
  and then re-encodes.
- `same_operator_ar` emits one token at a time. After each token it
  re-encodes the realized prefix, replaces only the semantic state, and
  applies the same retained `T_i` again. The prior, branch index, plane, and
  angle are not recomputed within the block.

Both policies recompile their operator at the next four-token boundary. No
gold continuation, posterior responsibility, sampling, beam search,
rejection, or post-generation correction participates.

## Fixed evaluation

- WikiText-103 validation split only
- seed `1337+2024`
- 64 fixed examples
- prompt length 64 tokens
- generated continuation length 64 tokens
- block width 4
- greedy decoding
- strict float32, TF32 disabled
- repository BPE tokenizer, vocabulary 8192

## Metrics and success criteria

Metrics are written to TSV. Decoded evidence and interpretation are written
to Markdown.

“Nearly identical” is registered as:

- same-prefix one-block token agreement at least `0.80`;
- same-prefix agreement at every horizon at least `0.70`;
- full-rollout token agreement at least `0.50`;
- block-4 distinct-2 within 10% relative of AR;
- block-4 immediate-repeat and period-4-repeat rates each within `0.05`
  absolute of AR; and
- collapsed-sample fraction within `0.10` absolute of AR.

The local comparison is the direct test of the homogeneous one-step law.
The 64-token comparison is stricter: one early greedy disagreement changes
all later prefixes, so failure there does not by itself prove that the local
law is mismatched.
