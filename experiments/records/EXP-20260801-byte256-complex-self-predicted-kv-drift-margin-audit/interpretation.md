# Interpretation

This file is generated from the preregistered read-only audit. See the TSV files for the complete measurements.

## Late checkpoint drift

| Pair | Raw cosine | Aligned cosine | Aligned rel. MSE | CKA |
|---|---:|---:|---:|---:|
| 500->750 | 0.984704 | 0.998848 | 0.018522 | 0.756046 |
| 750->1000 | 0.989308 | 0.999578 | 0.020379 | 0.862710 |

## Step-1000 decision margins

| H | Agreement | Match AR margin median | Mismatch AR margin median | Mismatch read cosine median | Mismatch read rel. MSE median |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.444092 | 0.672094 | 0.359903 | 0.992463 | 0.032442 |
| 3 | 0.164795 | 0.685605 | 0.563932 | 0.899272 | 16.184366 |
| 4 | 0.174561 | 0.834609 | 0.718475 | 0.918294 | 1.568300 |

## Target decoding

| H | Prior accuracy | Posterior accuracy | Canonical accuracy |
|---:|---:|---:|---:|
| 1 | 0.538330 | 0.109619 | 1.000000 |
| 2 | 0.070312 | 0.065430 | 1.000000 |
| 3 | 0.084473 | 0.069824 | 1.000000 |
| 4 | 0.065186 | 0.050293 | 1.000000 |

## Structural checks

- flip classification errors: 0
- max algebraic flip identity error: 7.629e-06
- minimum canonical target accuracy: 1.000000

## Findings

The online encoder does move in a way that is not merely one global rotation.
For steps 500->750 and 750->1000, orthogonal alignment raises mean cosine to
0.998848 and 0.999578, but aligned relative MSE remains 0.018522 and 0.020379
and centered CKA is only 0.756046 and 0.862710. The state norm also grows:
the earlier/later median norm ratios are 0.873919 and 0.862645.

This makes a coordinate-stabilization ablation scientifically justified, but
it does not make moving targets the primary explanation for the agreement
decline. The largest structural checkpoint motion occurs at steps 250->500
(CKA 0.479243 and aligned relative MSE 0.109524), while H2--H4 agreement stays
nearly flat at 0.362549->0.358154. Later, representation drift becomes much
smaller while agreement falls to 0.289307 and then 0.261149. Drift magnitude
therefore does not track the failure.

At H2, narrow decision boundaries contribute: mismatches have median AR
margin 0.359903 versus 0.672094 for matches, while the corresponding read
states remain close (median cosine 0.992463 and relative MSE 0.032442).

H3 and H4 are instead dominated by transition/self-composition error.
Mismatch margins remain ordinary (0.563932 and 0.718475), but median maximum
logit perturbations are 10.570817 and 12.077590. The H3/H4 mismatch read-state
relative MSE values are 16.184366 and 1.568300. Their errors are much too
large to characterize as only a sharp-boundary effect.

The strongest failure localization is the recurrent posterior itself. At H1,
the exact canonical future state decodes its observed byte with accuracy
1.000000 and the prior predicts it with accuracy 0.538330, but the
innovation-updated posterior decodes it with accuracy only 0.109619. This is
despite the parent run's H1 posterior/canonical cosine of about 0.995 and
relative MSE of about 0.013. Euclidean latent regression has therefore not
placed the posterior in the decoder-valid canonical basin required by the
next recurrence.

## EMA decision

A naive EMA future-state target is not the recommended next main experiment.
It would put the target in a lagged encoder coordinate system that the current
exact-inverse decoder does not invert, adding another coordinate mismatch.
The audit supports EMA only as a separately matched, coordinate-aligned
control.

The higher-priority mechanism test is to constrain the innovation posterior
behaviorally while continuing to keep it out of the current-token CE readout:
the prior remains the only current decoder input, while an auxiliary
posterior closure verifies that the recurrent state lies in the decoder basin
of its intended canonical future. This directly addresses the measured
0.109619 posterior recovery rather than only slowing encoder motion.
