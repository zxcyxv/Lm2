# Evaluation interpretation

All three policies started from the same 8,192 preserved validation block
boundaries and consumed no corpus token inside a block.

| policy | NLL | PPL-like | accuracy |
|---|---:|---:|---:|
| raw self-fed AR | 7.560595 | 1920.99 | 0.1099 |
| iterated T-K | 8.765177 | 6407.20 | 0.0200 |
| projected iterated T-K | 8.723647 | 6146.55 | 0.0128 |

Raw AR was the strongest of the three. Its h1 PPL-like was `89.99`, but
self-fed divergence raised h2--h4 to `1908.05`, `6485.33`, and `12229.22`.

Iterated T-K failed immediately: h1 PPL-like was `2383.26` and h1 accuracy
was `0.0627`. This agrees with the parent training record's failure of
corrected-state selected-token preservation. Lower canonical-state MSE did
not place `T(q)q` reliably in the decoder decision region.

Hard shell projection worsened h1 T-K PPL-like from `2383.26` to `4225.28`,
but lowered h2--h4 PPL-like from `8547.63/8874.70/9321.83` to
`6664.87/6818.68/7433.26`. Overall it lowered T-K NLL by `0.041530` and PPL
by 4.1%, while also lowering overall accuracy from `0.0200` to `0.0128`.
Thus it weakly improves the registered overall NLL comparison but does not
make the trajectory usable.

Generation equivalence failed decisively. Raw AR and iterated T-K agreed on
`0.0377` of tokens and `0.000244` of complete blocks. Raw AR and projected
T-K agreed on `0.0261` of tokens and no complete block. The projected-shell
structural error was `3.96e-6`, confirming that the intended hard projection
was applied.

These exponentiated losses are rollout PPL-like diagnostics, not standard
teacher-forced corpus perplexities.
