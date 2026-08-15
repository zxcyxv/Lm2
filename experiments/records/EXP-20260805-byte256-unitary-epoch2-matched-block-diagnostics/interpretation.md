# Matched sixteen-byte block interpretation

The comparison used the same 1,024 gold boundaries and 16,384 targets for all
three epoch-2 checkpoints. It does not assume that scan, H4, or H1 is driven by
logit scale.

| Mode | CE (T=1) | Best grid CE | Best T | Top-1 | Top-5 | Top-10 | Mean target rank | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 / chunk 1 | 7.063616 | 3.726457 | 4 | 0.183594 | 0.380493 | 0.544067 | 17.0071 | 0.294564 |
| H4 / chunk 4 | 3.494413 | 3.323052 | 2 | 0.203552 | 0.447205 | 0.646423 | 11.8530 | 0.332588 |
| Scan / chunk 16 | 2.912725 | 2.912725 | 1 | 0.235840 | 0.520203 | 0.732483 | 8.5833 | 0.377719 |

The scan advantage is not explained by a larger confidence margin: its mean
selected top-1/top-2 margin was the smallest (`0.7763`, versus H4 `1.0437`
and H1 `3.1410`). Temperature rescaling helped H1 and H4 substantially but did
not reverse any ordering; scan was already best at the unscaled temperature.
Its target ranks, MRR, and top-k accuracies were also best.

Greedy text quality answers a different question. Scan selected spaces on
82.75% of positions despite only 18.74% gold spaces, so its 16-byte argmax
samples are visually sparse. This is compatible with good target rank and NLL
when many correct alternatives receive probability but space narrowly wins.
It indicates a greedy-decoding/modal-bias problem, not evidence that the NLL
gain came from excessively large margins.
