# Training interpretation

The registered 1000-step run completed in `1540.4` seconds with peak CUDA
allocation `4.11 GB`. Validation raw-q NLL fell from `9.2091` to `6.1732`, so
the ordinary one-token CE path learned, but it was substantially slower than
the matched attached-MSE predecessor (`4.4673` at step 1000).

The stop-gradient self-redecode target remained intentionally sharp. Its
mean probability on the model's own selected token was `0.999995` at step
1000, and its top-1 action agreement was exactly `1.0` throughout.

The corrected distribution improved over the raw proposal relative to that
teacher: final forward KL was `0.527925`, versus raw-q KL `2.364059`, a ratio
of `0.223313`. However, the best corrected KL was the much earlier step-100
value `0.068715`. Corrected/teacher top-1 agreement likewise peaked near the
start and declined from `0.9960` at step 100 to `0.8416` at step 1000. The
moving shared CE/KL geometry therefore did not stabilize T's behavioral
collapse.

The output-level objective did not move corrected states toward the canonical
encoder state in Euclidean geometry. Final corrected canonical relative MSE
was `0.385217`, slightly worse than raw-q MSE `0.358870`. This is compatible
with the registered objective but confirms that decoder-visible matching and
canonical-state matching are not interchangeable in this run.

Corrected gold accuracy (`0.113922`) was close to raw-q accuracy (`0.112183`),
while corrected gold NLL was much worse (`9.091113` versus `6.173245`). The
student became sharper around its self-selected action without preserving a
well-calibrated corpus conditional distribution.

The primary raw-NLL decrease and corrected/raw KL-ratio gates passed. The
`0.90` corrected/teacher top-1 gate failed. The separately registered
alternating K--T rollout audit determines whether this one-step deficit also
compounds recurrently.
