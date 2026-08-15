# Step-4000 block-length interpretation

The trained block-4 policy gives the best balance between local readability
and diversity in this fixed-seed, 64-example diagnostic. Block 2 is not a
quality improvement: frequent re-anchoring and candidate resampling coincide
with strong two-token periodicity, 0.246976 exact adjacent-block repetition,
and 0.328125 collapse frequency.

Block 8 remains numerically viable despite lying outside the trained horizon.
It has the highest distinct-4, lowest measured collapse frequency, and higher
reference accuracy than blocks 2 and 4. This does not imply superior prose.
The samples contain substantial punctuation and function-word repetition,
and immediate and period-block repetition are both higher than at block 4.
The collapse detector is insensitive to alternating punctuation and other
short cycles.

Blocks 16 and 32 cross a clear stability boundary. Their immediate-repeat
rates approach 0.5, their mean longest identical runs exceed eight tokens,
and more than half of samples meet the registered collapse condition. The
current four-horizon training therefore supports, at most, limited
extrapolation to block 8; it does not support 16- or 32-token open-loop
generation.

The result uses one fixed sampled-noise seed. Absolute stochastic-policy
values can vary with candidate draws, so the robust claim is the qualitative
breakdown at blocks 16 and 32, not the exact ranking of blocks 4 and 8.
