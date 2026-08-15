# EXP-20260802 branch-normalized scaling/Jacobian audit

## Status

- State: completed
- Source: `../EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-normalized-residual-h16-ce-only-attached-stride16-13m/`
- Checkpoints: step 100, 300, 500, and 1000 (no step-750 artifact exists)
- Validation split only; test unread

## Questions

On fixed validation trajectories, are the following assumptions approximately
supported?

1. latent carrier norm grows as `sqrt(r)`;
2. residual direction change decays as `1/sqrt(r)`;
3. the unitary diagonal `D=diag(R,U)` behaves as a lossless highway in the
   complete local/product Jacobian rather than being cancelled by coupling;
4. the coupling perturbation `E=J-D` decays as `1/sqrt(r)`.

As a supplementary non-causal check, measure correlation between memory
Frobenius norm and per-row token CE. This cannot prove whether norm is or is
not task information.

## Protocol

- Restore step-100 and step-300 model weights.
- Use four fixed validation windows from seed 2336 and all 16 stride-16 roots.
- Record per-horizon latent norm, centered residual norm
  `||z_r-R^r z_0||`, memory Frobenius norm, correction norm,
  perpendicular correction ratio, and direction angle.
- Fit log-log exponents over horizons 2--16; hypotheses predict latent exponent
  `+0.5` and angle/perpendicular-ratio exponent `-0.5`.
- On the first fixed root, use four deterministic unit-norm JVP probes per
  horizon. Compute `Jv`, exact unitary `Dv`, `Ev=Jv-Dv`, their norms and
  alignment. Propagate each probe through all 16 actual local Jacobians and
  report final/input gain. These probes are directional evidence, not spectral
  norm guarantees.

Metrics are TSV and interpretation is Markdown. No training or checkpoint
mutation is performed.
