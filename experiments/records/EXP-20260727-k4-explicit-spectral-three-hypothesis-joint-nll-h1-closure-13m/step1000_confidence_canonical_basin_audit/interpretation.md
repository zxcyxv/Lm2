# Confidence-to-canonical-basin audit

- strong basin hypothesis supported by registered gates: **False**
- strong basin hypothesis falsified by registered gates: **True**
- initial gradient/residual cosine mean / median:
  `0.046831 /
  0.044453`
- positive / negative initial alignment:
  `0.839844 /
  0.160156`
- canonical B confidence mean / median / p95:
  `0.999777 /
  0.999999 /
  1.000000`
- fraction of canonical hB states with `p(B) >= 0.99`:
  `0.994141`
- proposal median confidence / relative distance:
  `0.143906 /
  0.026539`
- proposal-ascent median distance monotone decreasing: **False**
- observed `p(B) >= 0.99` states farther than their proposal:
  `937`

The audit tests the checkpoint's reachable decoder geometry. It can falsify
the claim that high B confidence identifies canonical hB, but finite
multi-start optimization cannot prove the universal absence of distant
high-confidence states.

## Interpretation

Canonical states are extremely confident token codes: median
`p(B|hB)=0.999999`, and 99.41% exceed 0.99. This confirms the forward
statement `hB -> B with high confidence`.

The converse fails. At the clean proposal, the median cosine between
`grad log p(B|u)` and the actual canonical residual `hB-u` is only 0.044.
Although 83.98% of cosines are weakly positive, 16.02% are negative and the
typical gradient is nearly orthogonal to the required projection direction.

From proposal initialization, ascent raises median B confidence:

| step | confidence | relative distance to hB |
|---:|---:|---:|
| 0 | 0.143906 | 0.027459 |
| 1 | 0.605352 | 0.027568 |
| 2 | 0.895019 | 0.027830 |
| 4 | 0.983475 | 0.028685 |
| 8 | 0.995666 | 0.030683 |
| 16 | 0.998146 | 0.034091 |
| 40 | 0.999161 | 0.037203 |

Thus confidence sharpening consistently moves away from canonical hB rather
than toward it. Among all registered trajectories, 946 observations reach
0.99 confidence; their median relative distance is 0.03588, and 937 are
farther from hB than their corresponding clean proposal.

The canonical basin is also highly anisotropic. A tangent perturbation around
hB with RMS matched to the ordinary proposal residual (about 2.7% relative
distance) drops median B confidence to 0.00354 and preserves B as top-1 in
only 46.09% of cases. Closeness to hB therefore does not itself imply high
confidence in arbitrary directions.

Random-sphere starts do not furnish a distant 99% counterexample in 40
steps—their median confidence reaches only 0.00375—so this finite audit does
not establish that arbitrarily remote high-confidence states are common.
The narrower conclusion is decisive for the proposed algebra: maximizing
B likelihood is not a canonical-state lift. It finds high-margin decoder
states roughly 3--4% from hB while the direction required for recurrent
canonical closure is almost orthogonal.

Consequently `softmax temperature/power sharpening` is valid in probability
space but cannot be pulled back to `hB` by following the B-likelihood
gradient in this checkpoint. A different quantity—direct projection
residual, full-distribution inverse geometry, or a learned low-rank
canonical correction—would be required to approximate the
decode--argmax--reencode map.
