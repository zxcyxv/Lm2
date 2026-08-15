# Interpretation

## Fixed known-future H16 audit

On the same eight validation contexts used at step 1000:

| Metric | Step 1000 | Step 6000 |
|---|---:|---:|
| Exact byte accuracy, 128 positions | 0.2500 | 0.2891 |
| Mean unique predicted bytes per 16 | 2.125 | 3.125 |
| Mean interference effective rank | 8.156 | 8.936 |
| H1 accuracy | 0.625 | 0.750 |
| H2 accuracy | 0.250 | 0.625 |
| H4 accuracy | 0.250 | 0.375 |
| H8 accuracy | 0.000 | 0.000 |
| H16 accuracy | 0.250 | 0.250 |

Training clearly improved the first few positions but did not carry the gain
through H16. Visible examples changed from mostly one initial byte plus spaces
to two-to-four plausible early bytes, followed again by spaces. For example,
gold `dge began with p` changed from step-1000 prediction sixteen spaces to
step-6000 `dg` followed by spaces; gold `t war against th` changed from `s`
plus spaces to `t in` plus spaces.

The additive checks remain numerical: maximum measurement reconstruction error
was `2.38e-7`, and maximum learned-R final-frame reconstruction error was
`7.63e-5`.

The interference map became broader, not more selective. At H16 the mean
absolute effective number of contributing writes increased from 13.70 to 14.94,
while the current-write absolute share increased only from 0.098 to 0.112.
Formal decomposition and effective rank therefore remain available, but the
map still resembles broad accumulation rather than a sparse sequence of
semantically distinct retrievals.

## 128-byte free generation

Across four fixed validation prompts:

| Policy | Mean exact byte accuracy | Mean unique bytes / 128 | Qualitative result |
|---|---:|---:|---|
| H1, re-encode every byte | 0.0781 | 24.25 | diverse but incoherent character/word fragments |
| H16, re-encode every block | 0.1738 | 6.00 | inflated by spaces; collapses after early bytes |

The H16 exact-match advantage is not better prose. Its samples repeatedly take
the form `o[spaces]f the[spaces]er...eee...`, whereas H1 emits varied but
ungrammatical strings such as `oor t9 ,,@ yalt@ft...`. Neither policy produces
coherent sentences at step 6000. The recurrent block has learned useful H1--H4
local continuation but not a meaningful sixteen-step thought trajectory.

These observations concern four long prompts and eight fixed one-block prompts;
they are direct samples, not a population-level human preference evaluation.

