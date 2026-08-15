# Step-1000 AR versus T-K versus projected-T-K block-4 evaluation

## Status

- State: preregistered before generation measurements
- Parent checkpoint: `step1000.pt`
- Expected checkpoint SHA-256:
  `4aadd63d831d1de6e391269f0ceb6dcd92da4870810e70e7158dd4abcd13b115`
- Validation-start SHA-256:
  `7dfbbbe6d3c4393b13459a4db116bbca9e429a5544e596c4035f7a140987e1e0`
- Split: validation only; test remains unread

## Question

Starting from identical gold four-token block boundaries, how do the
following target-independent generation policies compare?

1. `raw_ar`: T is absent. Compute `q=K(h)h`, decode q, greedily select one
   token, canonically re-encode that selected token, and repeat four times.
2. `iterated_tk`: compute `q_j=K(u_(j-1))u_(j-1)` and
   `u_j=T(q_j)q_j` four times without token feedback, then jointly decode the
   four-state tape.
3. `projected_iterated_tk`: use the same recurrence, but replace every
   `u_j` by its hard calibrated-shell projection before carrying it forward
   and jointly decoding the tape.

The initial state is the literal canonical boundary state in all policies.
Corpus future tokens are not consumed inside a block. They are read only
after all logits and greedy actions exist, for descriptive scoring.

## Fixed evaluation

- frozen step-1000 weights; no parameter update
- 128 preserved validation sequences
- context 256
- anchors 0, 4, ..., 252
- 8,192 independent blocks and 32,768 scored tokens
- greedy actions
- strict float32; TF32 disabled
- CUDA microbatch 4
- T is excluded from raw AR

## Metrics

- overall and horizon-specific held-out NLL, `exp(mean NLL)`, and accuracy
- token and exact-block agreement for every policy pair
- projected-shell structural error

The reported exponentiated loss is labeled `PPL-like`: raw AR conditions on
its own greedy history after h1, while both parallel policies remain latent
and token-free. It is not standard corpus teacher-forced perplexity.

The primary comparison is lower overall NLL. Generation equivalence requires
at least 0.90 token agreement and 0.75 exact-block agreement. Projection
improves iterated T-K only if its overall NLL is lower; otherwise it degrades
or leaves it unchanged. Numeric metrics are TSV and interpretation is
Markdown.
