# Interpretation

## 1. Density spectrum

The memory becomes somewhat more concentrated with horizon, but does not
collapse to a rank-one state.

| Horizon | Purity | Effective rank | Largest eigenvalue |
|---:|---:|---:|---:|
| H1 | 0.332 | 3.362 | 0.447 |
| H2 | 0.336 | 3.342 | 0.459 |
| H4 | 0.344 | 3.304 | 0.479 |
| H8 | 0.358 | 3.245 | 0.510 |
| H12 | 0.369 | 3.204 | 0.530 |
| H16 | 0.378 | 3.175 | 0.545 |

Thus late space collapse is accompanied by gradual spectral concentration, but
the normalized memory still carries roughly three effective singular modes at
H16. It is not principally a total-memory collapse into one pure mode.
`SS^dagger` and `S^dagger S` give the same nonzero normalized spectrum here, as
required. Maximum eigenvalue-sum error was `1.19e-7`.

## 2. Born measurement and exact interference

The complex write amplitudes are almost perfectly constructively aligned.
Negative pairwise Born cross terms are zero to numerical precision in nearly
all rows.

| Horizon | Born weight | Diagonal energy | Positive cross term | Cross / diagonal |
|---:|---:|---:|---:|---:|
| H1 | 0.01477 | 0.01477 | 0 | 0 |
| H2 | 0.01441 | 0.00721 | 0.00720 | 1.00 |
| H4 | 0.01189 | 0.00298 | 0.00891 | 2.99 |
| H8 | 0.00293 | 0.000373 | 0.00255 | 6.84 |
| H12 | 0.000565 | 0.0000581 | 0.000507 | 8.58 |
| H16 | 0.000971 | 0.0000743 | 0.000897 | 12.00 |

For identical aligned write amplitudes, cross/diagonal equals `H-1`. H2 and H4
are almost exactly at that limit, and H16 remains strongly coherent. This is
direct amplitude-level constructive interference, not the hidden-output signed
projection proxy used in the earlier audit. Maximum Born energy reconstruction
error was `1.86e-9`.

At H8--H16 all 72 audited predictions were spaces. Their mean density purity was
0.369 and effective rank 3.206, while cross/diagonal was 9.07. The observed
space attractor therefore coincides with a broadly accumulated, coherently
reinforced measurement mode rather than destructive cancellation or complete
memory rank collapse.

At the same time normalized Born weight falls by roughly an order of magnitude
from H1 to late horizons. The query selects a progressively smaller fraction of
the total memory norm even though the selected write amplitudes reinforce one
another. The most precise description is therefore:

> memory retains several modes, but the measurement branch locks its many write
> amplitudes into one weak, highly coherent common readout that the decoder maps
> to space.

## Relation to correct shallow bytes

Correct H1--H4 non-space predictions had mean Born weight 0.01430. H2 correct
non-space bytes had 0.01484 versus 0.01407 for H2 errors, but both groups had
the same near-maximal cross/diagonal ratio of 1.00. H3 correct bytes were all
spaces, and H4 had only one correct non-space byte.

Consequently constructive interference is already present in correct shallow
predictions, but it does not distinguish correct from incorrect decisions. It
appears to be the default accumulation law. With depth it becomes an
undifferentiated common mode instead of a content-selective interference
pattern.

These results support the algebraic Hilbert-space/Born interpretation of the
measurement machinery. They do not make the full network a physical quantum
channel and do not causally prove that positive cross terms cause the space
prediction; a phase-randomization or cross-term ablation would be required for
that claim.

