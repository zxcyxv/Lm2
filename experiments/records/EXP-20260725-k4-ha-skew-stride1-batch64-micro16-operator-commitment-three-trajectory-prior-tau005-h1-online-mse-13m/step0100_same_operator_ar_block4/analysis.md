# Interpretation

## Result

Step 100 does **not** satisfy the preregistered near-equivalence criterion.
The same-prefix one-block token agreement is `0.7109`, below `0.80`.
Although h1 agreement is exactly `1.0`, agreement falls to `0.5469`,
`0.6406`, and `0.6562` at h2--h4; all three later horizons miss the
per-horizon `0.70` threshold.

The exact four-token continuation agrees for `0.3594` of prefixes. This is
substantially stronger evidence than the old mismatched AR/block comparison,
but it is not AR equivalence.

## What h1 establishes

Both policies start from the same encoded state and apply the same retained
`T_i=Q_iK` at h1. Their exact h1 agreement verifies that the implementation
has removed the previous first-step/later-step code-path asymmetry.

The later mismatch therefore has a narrower interpretation:

`T_i^j hA`

is not yet decoded like repeated application of the same `T_i` to the
realized-prefix encoder state. It is open-loop latent error and encoder-state
re-anchoring discrepancy, rather than branch reselection or a different
transition law.

## Why the free-running agreement is not success

The 64-token token agreement is `0.7085`, above its registered `0.50`
threshold, but both policies have collapsed-sample fraction `1.0`.
Their average longest identical-token runs are `62.89` for AR and `61.88`
for block-4. The decoded examples are dominated by repetitions such as
`of of ...` and `to to ...`.

Consequently, much of the free-running agreement measures arrival at the
same token fixed point, not preservation of a coherent semantic trajectory.
The local one-block audit is the more informative result at this early
checkpoint.

The two repetition-rate gaps are small:

- immediate repeat differs by `0.0188`;
- period-4 repeat differs by `0.0130`; and
- collapsed fraction differs by `0.0`.

However, block-4 distinct-2 is `0.0389` versus AR's `0.0196`, almost a
two-fold ratio rather than the registered 10% neighborhood. This difference
also occurs inside an unusably collapsed regime and is not evidence that
block generation is better.

## Prior diagnostic

For these greedy prefixes, both policies selected branch 3 at every
four-token boundary. The prior entropy remains high (`~1.093` versus the
three-way maximum `log(3) ~= 1.099`), so a small logit preference is enough
to make argmax selection look completely collapsed. This does not contradict
the validation prior-winner accuracy metric, but it means step-100 greedy
generation does not yet exercise multiple committed operators.

## Scope

This audit is an intermediate checkpoint diagnosis, not the registered final
architecture test. It supports the structural claim that AR and block now
share one transition law, while rejecting the stronger claim that 100
updates have already made their generated trajectories nearly equivalent.
The same fixed protocol should be repeated at step 250 and step 1000 before
judging whether training closes the h2--h4 gap.
