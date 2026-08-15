# Interpretation

The same frozen step-500 checkpoint was evaluated on the same 8,192
validation block boundaries with no parameter update.

The registered raw alpha-zero path reached NLL `6.307470` and PPL `548.65`.
Enabling the calibrated hard shell post hoc raised NLL to `6.723399` and PPL
to `831.64`. The projected-minus-raw NLL was `+0.415929`, and projected PPL
was `1.516` times raw PPL. Under the preregistered classification this is
`projection_degrades`.

The degradation occurred at every horizon and was largest at the first:

| horizon | raw PPL | projected PPL | projected - raw NLL |
|---|---:|---:|---:|
| 1 | 224.68 | 682.93 | +1.111697 |
| 2 | 573.15 | 766.00 | +0.290044 |
| 3 | 790.38 | 927.49 | +0.159961 |
| 4 | 890.27 | 985.88 | +0.102016 |

Overall top-1 accuracy fell from `0.1050` to `0.0675`. Raw and projected
tokens agreed on `0.5182` of positions and on all four positions in only
`0.1050` of blocks.

The alpha-zero raw nesting error and projected shell-radius error were both
exactly zero, so the comparison exercised the intended raw and hard-shell
paths. The result is consistent with inference mismatch: the base model was
optimized through raw `K_A^j h_A` states, while its shell was only calibrated
before training, excluded from both optimizers, and never used by the
alpha-zero CE path. Hard radius replacement therefore destroys information
that the trained decoder uses, especially at horizon one.

This audit rejects post-hoc hard-shell projection as an inference improvement
for this checkpoint. It does not determine whether jointly training the same
model through the projected path would perform better.
