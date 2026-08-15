# Interpretation

The inference audit completed on 32 fixed validation prompts.  The causal
fixed-point implementation passed its correctness check: both initializers
became exactly equal to greedy AR by at most `n` iterations for every block
length `n`.

The latent orbit is a real but very small initializer advantage:

| Block | Initial latent agreement | Iterations to 99% | Iterations to exact |
|---:|---:|---:|---:|
| 4 | 25.00% | 3 | 3 |
| 8 | 14.84% | 7 | 7 |
| 16 | 7.42% | 15 | 15 |
| 32 | 3.71% | 30 | 31 |

The repeat-last control required 4, 8, 16, and 32 iterations for exact
agreement. Thus `K^n h_A` saves only one causal propagation iteration, while
the number of required iterations remains essentially linear in block
length.

This result falsifies the preregistered useful-speedup criterion for this
checkpoint and solver. Ordinary one-step CE learns the first latent successor
well enough that the first parallel proposal is usually correct, but its
open-loop powers do not pre-solve later AR dependencies. The simultaneous
causal update then advances the correct boundary roughly one token per
iteration, as expected for a triangular autoregressive fixed-point map.

The result is scoped to a fixed global linear `K` trained only on one-step CE.
It does not test a state-conditioned transition or a solver whose update
couples distant future positions spectrally.
