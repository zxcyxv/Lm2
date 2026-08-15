# Scaling and Jacobian audit

## Scope

These results use fixed validation trajectories at the saved step-100 and
step-300 checkpoints. Four directional JVP probes are not spectral-norm
estimates and cannot establish hard guarantees.

## Assumption 1: latent norm proportional to sqrt(r)

Not supported over H2--H16. The fitted log-log exponent was `0.0975` at step
100 and `0.0453` at step 300, rather than `+0.5`. Mean latent norm changed
from `78.28` to `96.71` across H1--H16 at step 100 and from `170.98` to
`189.53` at step 300. The carrier was dominated by its already-large root,
so the proposed random-walk scaling did not describe this range.

## Assumption 2: direction change proportional to 1/sqrt(r)

Only approximately supported at step 100 and not as a fixed law. Direction
angle exponent was `-0.4406` at step 100, close to `-0.5`, but `-1.0268` at
step 300. The perpendicular correction-ratio exponents were correspondingly
`-0.4438` and `-1.0332`. Training changed the decay law substantially.

## Assumption 3: unitary D is a lossless complete-gradient highway

Not supported as a statement about the complete Jacobian. Exact `Dv` retained
the incoming tangent norm at every local step, but the full 16-step JVP gain
was `8.22--9.21` at step 100 and `6.00--6.34` at step 300. Thus the graph
contains an exact unitary diagonal component, while the summed complete
derivative is strongly amplified rather than norm-preserving. Late local
`Jv` became nearly aligned with `Dv`; this does not undo amplification already
accumulated in the tangent.

## Assumption 4: coupling E decays as 1/sqrt(r)

Supported in direction but with a different exponent. Mean `||Ev||/||Dv||`
fell from `0.610` at H1 to `0.061` at H16 at step 100, and from `0.393` to
`0.0667` at step 300. Fitted exponents were `-1.0646` and `-0.8662`, faster
than `-0.5`. Normalized coupling weakened with horizon on these probes, but
its early-step contribution was sufficient to produce a large product gain.

## Memory norm supplement

Memory Frobenius norm grew approximately linearly, with exponent `1.0349` at
step 100 and `1.0692` at step 300, rather than as sqrt(r). This is consistent
with strongly correlated writes. Pearson correlation between memory norm and
per-row token CE was small positive (`0.0565`, `0.0578`). This supplies little
linear predictive evidence in either direction and does not establish that
memory norm is information-free.

## Late checkpoints and corrected centered carrier

Continuing optimization changed the four-assumption picture as follows:

| Step | raw-z exponent | centered `z-R^r z0` exponent | angle exponent | E/D exponent | mean H16 product gain |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.098 | 0.883 | -0.441 | -1.065 | 8.801 |
| 300 | 0.045 | 0.643 | -1.027 | -0.866 | 6.177 |
| 500 | 0.033 | 0.599 | -1.314 | -0.737 | 4.531 |
| 1000 | 0.024 | 0.557 | -1.450 | -0.594 | 3.364 |

The corrected centered residual carrier approached the proposed `sqrt(r)`
law, reaching exponent `0.557` at step 1000. Thus the first assumption was
wrong for raw latent norm but became approximately supported for the quantity
the random-walk argument actually concerns.

Direction change did not approach `r^-1/2`; it decayed progressively faster,
to approximately `r^-1.45`. Coupling E/D did approach the proposed exponent,
moving from `-1.065` to `-0.594`. Complete H16 directional gain steadily fell
from `8.80` to `3.36`, but remained far from lossless gain one. Therefore late
training supports centered sqrt growth and inverse-sqrt coupling, rejects a
fixed inverse-sqrt direction-angle law, and shows improvement but not
confirmation of complete-Jacobian norm preservation.
