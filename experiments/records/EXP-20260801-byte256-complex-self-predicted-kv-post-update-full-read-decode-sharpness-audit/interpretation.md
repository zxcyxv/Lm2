# Interpretation

H1 is the primary basin test. H2--H4 are supplementary rollout diagnostics.

## H1 basin test

| Group | Rows | P target acc. | P/A top-1 | P/Znext top-1 | Innovation enters P basin | Innovation leaves P basin | P max prob. | Znext max prob. | Znext >=99% | Znext target acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 4096 | 0.584229 | 0.151123 | 0.400879 | 0.305176 | 0.055420 | 0.594960 | 0.460469 | 0.130615 | 0.338867 |
| p_correct | 2393 | 1.000000 | 0.162975 | 0.554952 | 0.429586 | 0.037610 | 0.767516 | 0.574325 | 0.210196 | 0.554952 |
| p_incorrect | 1703 | 0.000000 | 0.134469 | 0.184381 | 0.130358 | 0.080446 | 0.352491 | 0.300484 | 0.018790 | 0.035232 |

## Supplementary horizons

| H | P/Znext top-1 | P max prob. | Znext max prob. | Znext >=99% | P entropy | Znext entropy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.400879 | 0.594960 | 0.460469 | 0.130615 | 1.391463 | 2.474438 |
| 2 | 0.197510 | 0.367204 | 0.434178 | 0.045654 | 2.175142 | 2.294475 |
| 3 | 0.133301 | 0.398613 | 0.464258 | 0.025879 | 1.992066 | 2.005875 |
| 4 | 0.140381 | 0.471113 | 0.509825 | 0.020752 | 1.705429 | 1.753504 |

## Calibration and scope

- canonical target top-1 errors: 0
- canonical mean max-softmax probability: 0.999973
- no matched no-MSE model was trained; this audit measures the basin reached by the MSE-trained model, not causal improvement over no-MSE training.

## Conclusion

The H1 innovation does move states toward P's selected-token basin. Before the
current innovation, only `15.11%` of local A decodes shared P's top-1. After
the innovation, `40.09%` of Znext decodes did. Basin entries (`30.52%`)
substantially exceeded exits (`5.54%`). This is a real positive effect of the
trained update at inference, although a matched no-MSE training control would
still be required to attribute that effect causally to MSE rather than the
joint training setup as a whole.

The intended future-hidden interpretation nevertheless fails at the absolute
level:

- P/Znext top-1 agrees on only `40.09%` of H1 rows.
- Znext is less sharp than P: mean max probability is `0.460469` versus
  `0.594960`, and entropy is `2.474438` versus `1.391463`.
- Only `13.06%` of Znext rows reach 99% max probability, while a canonical
  encoder hidden reaches mean confidence `99.9973%` with exact top-1 recovery.
- Even among the 1,642 P/Znext-matching rows, Znext mean max probability is
  `0.765069` versus P's `0.824769`, and only `32.40%` reach 99%.

Conditioning on P being correct makes the objective interaction explicit. If
P already selects observed B, Znext remains in that same correct basin only
`55.50%` of the time. If P is wrong, Znext follows P on `18.44%` of rows but
recovers observed B on only `3.52%`; it is usually diffuse rather than sharply
choosing either competing basin.

Thus the MSE-trained transition provides a partial basin-entry correction, but
Znext cannot currently be treated as the realized future encoder hidden. H2--
H4 reinforce this conclusion but remain supplementary rather than the primary
test.
