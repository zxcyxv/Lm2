# Inference-time parallel solver audit

## Status

- State: preregistered before evaluation
- Checkpoint: step-1000 `inverse_rms` checkpoint from the parent experiment
- Test split remains unread

## Question

Starting from a parallel `K^1 h_A ... K^n h_A` latent decode, how many
simultaneous causal fixed-point updates are needed to reproduce the same
model's deterministic greedy AR tokens?

## Fixed protocol

- validation split, seed `1337 + 7270`
- 32 prompts of length 64
- block lengths 4, 8, 16, and 32
- target: literal greedy AR generation from the same checkpoint
- initialization:
  - latent orbit: jointly decode `K^1 h_A ... K^n h_A`
  - control: repeat the prompt's final token
- one solver iteration evaluates all block positions in one teacher-forced
  causal pass and simultaneously replaces every candidate by its argmax
- maximum iterations equal the block length

## Metrics and success criteria

For every initializer, block length, and iteration, report token agreement
with greedy AR and exact-block agreement. Also report the first iteration at
which mean token agreement reaches 95%, reaches 99%, and all sampled blocks
are exactly equal.

The hypothesis is supported if the latent initializer reaches 99% agreement
in fewer iterations than the block length and no slower than the repeat-token
control. Exact equality by `n` iterations is a causal-triangular correctness
check, not evidence of speedup.
