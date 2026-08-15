# Interpretation

Step 6000 gives the following comparison with the preserved step-1000 audit.

| Assumption | Step 1000 | Step 6000 | Verdict at 6000 |
|---|---:|---:|---|
| centered residual exponent, target `+0.5` | +0.557 | +0.404 | approximately supported |
| direction-angle exponent, target `-0.5` | -1.450 | -1.327 | not supported; decays much faster |
| mean H16 directional Jacobian gain, target near `1` | 3.364 | 3.852 | not supported; amplifying |
| `E/D` exponent, target `-0.5` | -0.594 | -0.627 | approximately supported |

The first and fourth structural scaling claims remain approximate. The second
does not become a `1/sqrt(r)` law with longer training. The unitary diagonal
still supplies an explicit isometric component, but the complete coupled
Jacobian is not a lossless near-unit map: all four measured H16 product gains
were 3.554--4.033. Thus assumption 3 remains false and is slightly worse than
at step 1000.

Mean local `E/D` was 0.1355 versus 0.1245 at step 1000. Its horizon decay fits
the predicted exponent, but its overall magnitude did not shrink with training.
Memory Frobenius growth remained near-linear (exponent 1.052), not square-root,
consistent with correlated recurrent writes. Memory norm/token-CE Pearson
correlation rose from 0.104 to 0.206 but remains supplementary and non-causal.

